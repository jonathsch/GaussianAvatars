from pathlib import Path
from typing import Optional, Iterable
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import yaml

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import tyro
import wandb
from torchmetrics import MeanAbsoluteError
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
    LearnedPerceptualImagePatchSimilarity,
)
from torchvision.utils import save_image

from scene.goliath_gaussian_model import GoliathGaussianModel
from mesh_renderer import NVDiffRenderer
from datasets.goliath import GoliathHeadDataset, collate_fn, worker_init_fn
from gaussian_renderer import render_gsplat
from utils.vector_ops import to_hvec
from torch.profiler import profile, record_function, ProfilerActivity
from utils.camera_utils import get_camera_trajectory, get_light_path
import av
import gsplat
from slang.ops import unproject_depth
from slang.pbr import pbr_bsdf

logger = logging.getLogger(__name__)


@dataclass
class GoliathDatasetConfig:
    # Path to the dataset
    root_path: Path
    # Static asset path
    shared_assets_path: Path
    # Load only fully lit tracking frames
    fully_lit_only: bool = False
    # Load only partially lit frames
    partially_lit_only: bool = False
    # Load only a specific segment
    segment: Optional[str] = None
    # Load only a subset of the frames
    frames_subset: Optional[Iterable[int]] = None
    # Load only a subset of the cameras
    cameras_subset: Optional[Iterable[str]] = None
    # Downsample all image-related data by this factor
    downsample_factor: int = 1


@dataclass
class GaussianAvatarConfig:
    # 3D Gaussians
    iterations = 600_000  # 30_000 (original)
    position_lr_init = (
        0.005  # (scaled up according to mean triangle scale)  #0.00016 (original)
    )
    position_lr_final = (
        0.00005  # (scaled up according to mean triangle scale) # 0.0000016 (original)
    )
    position_lr_delay_mult = 0.01
    position_lr_max_steps = 600_000  # 30_000 (original)
    feature_lr = 0.0025
    opacity_lr = 0.05
    scaling_lr = (
        0.017  # (scaled up according to mean triangle scale)  # 0.005 (original)
    )
    rotation_lr = 0.001
    densification_interval = 2_000  # 100 (original)
    opacity_reset_interval = 60_000  # 3000 (original)
    densify_from_iter = 10_000  # 500 (original)
    densify_until_iter = 600_000  # 15_000 (original)
    densify_grad_threshold = 0.0002

    # GaussianAvatars
    flame_expr_lr = 1e-3
    flame_trans_lr = 1e-6
    flame_pose_lr = 1e-5
    percent_dense = 0.01
    lambda_dssim = 0.2
    lambda_xyz = 1e-2
    threshold_xyz = 1.0
    metric_xyz = False
    lambda_scale = 1.0
    threshold_scale = 0.6
    metric_scale = False
    lambda_dynamic_offset = 0.0
    lambda_laplacian = 0.0
    lambda_dynamic_offset_std = 0  # 1.


@dataclass
class GoliathGaussianAvatarConfig:
    # Dataset configuration
    data: GoliathDatasetConfig

    # Model configuration
    model: GaussianAvatarConfig

    # Path to the .pt file. If provide, it will skip training and render a video
    ckpt: Optional[str] = None

    # Maximum number of training steps
    max_steps: int = 600_000

    # Relative strength of the SSIM loss
    lambda_ssim: float = 0.2
    lambda_xyz: float = 1e-2
    threshold_xyz: float = 1.0

    # Where to save resuls
    output_dir: Path = Path("output/gaussian_avatar_2dgs_goliath")
    # Enable wandb logging
    use_wandb: bool = False
    # Log information to wandb every n steps
    log_interval: int = 100
    # Run and save evaluation every n steps
    eval_interval: int = 20_000


class Runner:
    def __init__(self, config: GoliathGaussianAvatarConfig):
        self.cfg = config
        self.device = torch.device("cuda:0")

        # Create output directories
        self.log_dir = config.output_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir.mkdir(parents=True, exist_ok=False)
        self.log_media_dir = self.log_dir / "media"
        self.log_media_dir.mkdir()
        if config.use_wandb:
            wandb.init(
                dir=config.output_dir.absolute(),
                project="rga",
                group="gaussian_avatar_2dgs_goliath",
                config=asdict(config),
                tags=["2dgs", "goliath"],
            )

        # Dump config
        with open(self.log_dir / "config.yaml", "w") as f:
            yaml.dump(asdict(config), f)

        self.gaussians = GoliathGaussianModel(
            3, config.data.root_path / "kinematic_tracking" / "template_mesh.obj"
        )
        self.mesh_renderer = NVDiffRenderer()

        if config.ckpt:
            (model_params, first_iter) = torch.load(config.ckpt)
            self.gaussians.restore(model_params, config)

        self.train_set = GoliathHeadDataset(**asdict(config.data), split="train")
        self.val_set = GoliathHeadDataset(**asdict(config.data), split="test")

        # Losses and metrics
        self.mae = MeanAbsoluteError().to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=(0.0, 1.0)).to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=(0.0, 1.0)).to(
            self.device
        )
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(
            self.device
        )

    def render(
        self,
        vertices: torch.Tensor,
        world_to_cam: torch.Tensor,
        Ks: torch.Tensor,
        width: int,
        height: int,
        light_pos: Optional[torch.Tensor] = None,
        light_intensity: Optional[torch.Tensor] = None,
        bg_img: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        self.gaussians.update_mesh_properties(vertices)

        means3d = self.gaussians.get_xyz
        opacity = self.gaussians.get_opacity
        scales = self.gaussians.get_scaling
        rotations = self.gaussians.get_rotation
        shs = self.gaussians.get_features
        pbr_material = self.gaussians.get_pbr_material

        (
            render_pbr,
            render_alpha,
            render_normals,
            normals_from_depth,
            render_dist,
            render_median,
            meta,
        ) = gsplat.rasterization_2dgs(
            means=means3d,
            quats=rotations,
            scales=scales,
            opacities=opacity.squeeze(-1),
            colors=pbr_material,
            viewmats=world_to_cam,
            Ks=Ks,
            width=width,
            height=height,
            render_mode="RGB+ED",
        )

        # Shade
        c2w = torch.linalg.inv(world_to_cam)
        kd = render_pbr[..., :3]
        ks = render_pbr[..., 3:6]
        depth = render_pbr[..., 6:7]
        pos = unproject_depth(depth, cam_to_world=c2w, projection=Ks)
        nrm = render_normals / render_alpha.detach().clamp_min(1e-5)  # [N, H, W, 3]

        campos = c2w[:, :3, 3]  # [C, 3]
        light_pos = light_pos.cuda()
        light_intensity = light_intensity.cuda()
        render_rgb = pbr_bsdf(kd, ks, pos, nrm, campos, light_pos, light_intensity)

        if bg_img is not None:
            render_rgb = render_rgb * render_alpha + bg_img * (1 - render_alpha)

        meta["gradient_2dgs"].retain_grad()
        radii = torch.zeros(
            means3d.shape[0], device=meta["radii"].device, dtype=meta["radii"].dtype
        )
        radii[meta["gaussian_ids"]] = meta["radii"]

        return {
            "render": render_rgb,
            "viewspace_points": meta["gradient_2dgs"],
            "visibility_filter": radii > 0,
            "radii": radii,
            "render_depth": depth,
            "render_alpha": render_alpha,
            "render_normals": render_normals,
            "normals_from_depth": normals_from_depth,
            "render_dist": render_dist,
            "render_median": render_median,
            "kd": kd,
            "ks": ks,
            "meta": meta,
        }

    @torch.no_grad()
    def render_trajectory(
        self, K: torch.Tensor, step: int, width: int, height: int, num_frames: int = 96
    ):
        """
        Render a trajectory around the head and save as video.

        Args:
            K: Camera intrinsics [1, 3, 3]
            step: Current training step
            width: Render width
            height: Render height
            num_frames: Number of frames to render
        """
        cam_path = get_camera_trajectory(num_frames=num_frames)  # [N, 4, 4]
        cam_path = torch.from_numpy(cam_path).to(K)

        light_path = get_light_path(num_frames=num_frames)  # [N, 3]
        light_path = torch.from_numpy(light_path).to(K)
        light_intensity = torch.tensor(3.0, device=self.device).reshape(1, 1, 1)

        container = av.open(str(self.log_media_dir / f"eval_{step:06d}.mp4"), "w")
        stream = container.add_stream("mpeg4", rate=24)
        stream.width = 4 * width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.bit_rate = 4 * 8000000

        for i, (w2c, l_pos) in enumerate(zip(cam_path, light_path)):
            # Get vertices of i-th timestep
            vertices = self.val_set.get_vertices_by_timestep(i).cuda()  # [V, 3]

            render_pkg = self.render(
                vertices=vertices,
                world_to_cam=w2c.unsqueeze(0),
                Ks=K.unsqueeze(0),
                width=width,
                height=height,
                light_pos=l_pos.view(1, 1, 3),
                light_intensity=light_intensity,
            )

            image = render_pkg["render"].squeeze().cpu().numpy()  # [H, W, 3]
            kd = render_pkg["kd"].squeeze().cpu().numpy()  # [H, W, 3]
            ks = render_pkg["ks"].squeeze().cpu().numpy()
            normals = (
                render_pkg["render_normals"].squeeze().cpu().numpy() * 0.5 + 0.5
            )  # [H, W, 3]

            canvas = np.concatenate([image, kd, ks, normals], axis=1) * 255
            canvas = canvas.clip(0, 255).astype(np.uint8)

            frame = av.VideoFrame.from_ndarray(canvas, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)

        container.close()

    @torch.no_grad()
    def evaluate(self, step: int):
        dataloader_val = DataLoader(
            self.val_set,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            worker_init_fn=worker_init_fn,
        )

        self.psnr.reset()
        self.ssim.reset()
        self.lpips.reset()

        render_height, render_width = self.val_set.get_image_size()

        gt_imgs, pred_imgs = [], []
        for idx, batch in enumerate(iter(dataloader_val)):
            if batch is None:
                continue

            # Get mesh vertices from head to world space
            head_pose = batch["head_pose"]
            head_pose_4x4 = torch.cat(
                [head_pose, torch.zeros_like(head_pose[:, :1, :])], dim=1
            )
            head_pose_4x4[:, 3, 3] = 1.0
            w2c = batch["world_to_cam"] @ head_pose_4x4

            world_to_cam = w2c
            world_to_cam = world_to_cam.cuda()
            K = batch["K"].cuda()

            verts = batch["vertices"].squeeze(0).cuda()
            l_pos = (batch["light_pos"] - head_pose[:, :3, 3]) @ head_pose[:, :3, :3]
            lint = batch["light_intensity"].cuda()

            background = batch["background"].cuda()
            fully_lit = batch["is_fully_lit_frame"].cuda()
            background = background * fully_lit[:, None, None, None].float()
            background = background.permute(0, 2, 3, 1)[..., :3]

            render_pkg = self.render(
                verts,
                world_to_cam,
                K,
                render_width,
                render_height,
                light_pos=l_pos,
                light_intensity=lint,
                bg_img=background,
            )

            pred_img = render_pkg["render"].permute(0, 3, 1, 2).contiguous()
            if pred_img.isnan().any():
                continue

            pred_img = pred_img.clamp(0.0, 1.0)
            gt_img = batch["image"].cuda()

            self.mae.update(pred_img, gt_img)
            self.psnr.update(pred_img, gt_img)
            self.ssim.update(pred_img, gt_img)
            self.lpips.update(pred_img, gt_img)

            if idx % 50 == 0:
                gt_imgs.append(gt_img[0].permute(1, 2, 0).cpu().numpy())
                pred_imgs.append(pred_img[0].permute(1, 2, 0).cpu().numpy())

            if idx > 250:
                break

        print(
            f"Step {step}: L1: {self.mae.compute()}, PSNR: {self.psnr.compute()}, SSIM: {self.ssim.compute()}, LPIPS: {self.lpips.compute()}"
        )
        if self.cfg.use_wandb:
            gt_imgs = np.concatenate(gt_imgs, axis=1)
            pred_imgs = np.concatenate(pred_imgs, axis=1)

            wandb.log(
                {
                    "val/l1": self.mae.compute().item(),
                    "val/psnr": self.psnr.compute().item(),
                    "val/ssim": self.ssim.compute().item(),
                    "val/lpips": self.lpips.compute().item(),
                    "image/gt": wandb.Image(gt_imgs, file_type="jpg"),
                    "image/pred": wandb.Image(pred_imgs, file_type="jpg"),
                },
                step=step,
            )

    def train(self):
        trainloader = DataLoader(
            self.train_set,
            batch_size=1,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=8,
            prefetch_factor=8,
            worker_init_fn=worker_init_fn,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        self.gaussians.create_from_pcd(None, 1.0)
        self.gaussians.training_setup(config.model)

        render_height, render_width = self.train_set.get_image_size()

        pbar = tqdm(range(0, self.cfg.max_steps))
        for iteration in pbar:
            # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            self.gaussians.update_learning_rate(iteration)

            # Increase SH level
            if iteration % 1_000 == 0:
                self.gaussians.oneupSHdegree()

            try:
                batch = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                batch = next(trainloader_iter)

            if batch is None:
                continue

            # Get mesh vertices from head to world space
            head_pose = batch["head_pose"]
            head_pose_4x4 = torch.cat(
                [head_pose, torch.zeros_like(head_pose[:, :1, :])], dim=1
            )
            head_pose_4x4[:, 3, 3] = 1.0
            w2c = batch["world_to_cam"] @ head_pose_4x4

            world_to_cam = w2c
            world_to_cam = world_to_cam.cuda()
            K = batch["K"].cuda()

            verts = batch["vertices"].squeeze(0).cuda()
            l_pos = (batch["light_pos"] - head_pose[:, :3, 3]) @ head_pose[:, :3, :3]
            lint = batch["light_intensity"].cuda()

            background = batch["background"].cuda()
            fully_lit = batch["is_fully_lit_frame"].cuda()
            background = background * fully_lit[:, None, None, None].float()
            background = background.permute(0, 2, 3, 1)[..., :3]

            render_pkg = self.render(
                verts,
                world_to_cam,
                K,
                render_width,
                render_height,
                light_pos=l_pos,
                light_intensity=lint,
                bg_img=background,
            )

            image, viewspace_point_tensor, visibility_filter, radii = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )
            image = image.squeeze().permute(2, 0, 1)
            viewspace_point_tensor.retain_grad()

            if image.isnan().any():
                print("NaN in image", flush=True)
                continue

            # Loss
            gt_img = batch["image"].cuda().squeeze(0)

            l1_loss = torch.nn.functional.l1_loss(image, gt_img) * (
                1.0 - self.cfg.lambda_ssim
            )

            ssim_loss = (
                1.0 - self.ssim(image[None, ...], gt_img[None, ...])
            ) * self.cfg.lambda_ssim

            if self.cfg.model.lambda_scale != 0:
                if self.cfg.model.metric_scale:
                    scale_loss = (
                        torch.nn.functional.relu(
                            self.gaussians.get_scaling[visibility_filter]
                            - self.cfg.model.threshold_scale
                        )
                        .norm(dim=1)
                        .mean()
                        * self.cfg.model.lambda_scale
                    )
                else:
                    # losses['scale'] = F.relu(gaussians._scaling).norm(dim=1).mean() * opt.lambda_scale
                    scale_loss = (
                        torch.nn.functional.relu(
                            torch.exp(self.gaussians._scaling[visibility_filter])
                            - self.cfg.model.threshold_scale
                        )
                        .norm(dim=1)
                        .mean()
                        * self.cfg.model.lambda_scale
                    )

            xyz_loss = (
                torch.nn.functional.relu(
                    self.gaussians._xyz[visibility_filter].norm(dim=-1)
                    - self.cfg.threshold_xyz
                ).mean()
                * self.cfg.lambda_xyz
            )

            loss = l1_loss + ssim_loss + xyz_loss + scale_loss
            loss.backward()

            # Logging
            if iteration % 10 == 0:
                desc = f"l1={l1_loss.item():.3f}, ssim={ssim_loss.item():.3f}, xyz={xyz_loss.item():.3f}, num_gaussians={self.gaussians.get_xyz.shape[0]}"
                pbar.set_description(desc)

            if self.cfg.use_wandb and iteration % self.cfg.log_interval == 0:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/l1_loss": l1_loss.item(),
                        "train/ssim_loss": ssim_loss.item(),
                        "train/xyz_loss": xyz_loss.item(),
                        "train/num_gs": self.gaussians.get_xyz.shape[0],
                    },
                    step=iteration,
                )

            # Densification
            if iteration < self.cfg.model.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )

                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter, meta=render_pkg["meta"]
                )

                if (
                    iteration > self.cfg.model.densify_from_iter
                    and iteration % self.cfg.model.densification_interval == 0
                ):
                    size_threshold = (
                        20
                        if iteration > self.cfg.model.opacity_reset_interval
                        else None
                    )
                    self.gaussians.densify_and_prune(
                        self.cfg.model.densify_grad_threshold,
                        0.005,
                        self.train_set.get_camera_extend(),
                        size_threshold,
                    )

                if (iteration + 1) % self.cfg.model.opacity_reset_interval == 0:
                    self.gaussians.reset_opacity()

            # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
            # quit()

            # Optimize
            self.gaussians.optimizer.step()
            self.gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration % self.cfg.eval_interval == 0:
                self.evaluate(iteration)
                self.render_trajectory(
                    K=K[0],
                    step=iteration,
                    width=render_width,
                    height=render_height,
                    num_frames=96,
                )

        # Final evaluation
        self.evaluate(iteration)
        self.render_trajectory(iteration)


if __name__ == "__main__":
    config = tyro.cli(GoliathGaussianAvatarConfig)
    runner = Runner(config)
    runner.train()
