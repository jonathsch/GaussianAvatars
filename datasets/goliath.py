from typing import Dict, List, Tuple, Any, Optional, Iterable, Set
import json
import zipfile
from pathlib import Path
from functools import lru_cache
from PIL import Image
import pillow_avif  # noqa
from io import BytesIO

import numpy as np
import trimesh
import pandas as pd
import torch
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import to_tensor
import trimesh.exchange
import trimesh.exchange.load
import trimesh.exchange.ply

CACHE_LENGTH = 160


class GoliathHeadDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_path: Path,
        shared_assets_path: Path,
        split: str,
        fully_lit_only: bool = False,
        partially_lit_only: bool = False,
        segment: Optional[str] = None,
        frames_subset: Optional[Iterable[int]] = None,
        cameras_subset: Optional[Iterable[str]] = None,
        downsample_factor: int = 1,
    ):
        super().__init__()

        if split not in {"train", "test"}:
            raise ValueError(f"Invalid split: {split}. Options are 'train' and 'test'")
        self.root_path: Path = Path(root_path)
        self.shared_assets_path: Path = shared_assets_path
        self.split: str = split
        self.fully_lit_only: bool = fully_lit_only
        self.partially_lit_only: bool = partially_lit_only

        self.downsample_factor = downsample_factor

        # Get list of cameras after filtering
        self.cameras_subset = set(cameras_subset or {})
        self.cameras = list(self.load_camera_calibration().keys())

        self.frames_subset = set(frames_subset or {})
        self.frames_subset = set(map(int, self.frames_subset))

        self.segment = segment

        self.frame_list = self.load_frame_list(
            fully_lit_only=fully_lit_only,
            partially_lit_only=partially_lit_only,
            segment=segment,
        )

    @lru_cache(maxsize=1)
    def load_shared_assets(self) -> Dict[str, Any]:
        return torch.load(self.shared_assets_path, map_location="cpu")

    def asset_exists(self, frame: int) -> bool:
        return frame in self.get_fully_lit_frames()

    @lru_cache(maxsize=1)
    def get_image_size(self) -> Tuple[int, int]:
        frame = self.frame_list[0]
        camera = self.camera_ids[0]
        img = self.load_image(frame, camera)
        return img.shape[1:]

    @lru_cache(maxsize=1)
    def load_camera_calibration(self):
        with open(self.root_path / "camera_calibration.json", "r") as f:
            camera_calibration = json.load(f)["KRT"]
        camera_params = {str(c["cameraId"]): c for c in camera_calibration}

        # We might have images for fewer cameras than there are listed in the json file
        image_zips = set(
            [x for x in (self.root_path / "image").iterdir() if x.is_file()]
        )
        image_zips = set([x.name.split(".")[0][3:] for x in image_zips])
        camera_params = {
            cid: cparams for cid, cparams in camera_params.items() if cid in image_zips
        }

        return camera_params

    @lru_cache(maxsize=1)
    def get_camera_parameters(self, camera_id: str) -> Dict[str, Any]:
        krt = self.load_camera_calibration()[camera_id]

        K = np.array(krt["K"], dtype=np.float32).T
        K[:2, :2] /= 2 * self.downsample_factor
        K[:2, 2] = (K[:2, 2] + 0.5) / (2 * self.downsample_factor) - 0.5

        Rt = np.array(krt["T"], dtype=np.float32).T
        Rt[:3, 3] /= 1000.0  # NOTE: Convert to meters
        R, t = Rt[:3, :3], Rt[:3, 3]
        focal = np.array(K[:2, :2], dtype=np.float32)
        princpt = np.array(K[:2, 2], dtype=np.float32)

        return {
            "world_to_cam": Rt,
            "K": K,
            "campos": R.T.dot(-t),
            "camrot": R,
            "focal": focal,
            "princpt": princpt,
        }

    @property
    def camera_ids(self) -> List[str]:
        return self.cameras

    def filter_frame_list(self, frame_list: List[int]) -> List[int]:
        frames = frame_list
        if self.frames_subset:
            frames = list(set(frame_list).intersection(self.frames_subset))
        return frames

    @lru_cache(maxsize=2)
    def load_frame_list(
        self,
        fully_lit_only: bool = False,
        partially_lit_only: bool = False,
        segment: Optional[str] = None,
    ) -> List[int]:
        assert not (fully_lit_only and partially_lit_only)

        df = pd.read_csv(self.root_path / "frame_splits_list.csv")
        frame_list = df[df.split == self.split].frame.tolist()

        if segment:
            segment_df = pd.read_csv(self.root_path / "frame_segments_list.csv")
            segment_frames = segment_df[segment_df.segment_name == segment].frame.tolist()
            frame_list = list(set(frame_list).intersection(segment_frames))

        if not (fully_lit_only or partially_lit_only or segment):
            frame_list = list(frame_list)
            return self.filter_frame_list(frame_list)

        if fully_lit_only:
            fully_lit = {
                frame for frame, index in self.load_light_pattern() if index == 0
            }
            frame_list = [f for f in fully_lit if f in frame_list]
            return self.filter_frame_list(frame_list)
        else:
            light_pattern = self.load_light_pattern_meta()["light_patterns"]
            # NOTE: it only filters the frames with 5 lights on
            partially_lit = {
                frame
                for frame, index in self.load_light_pattern()
                if len(light_pattern[index]["light_index_durations"]) == 5
            }
            frame_list = [f for f in partially_lit if f in frame_list]
            return self.filter_frame_list(frame_list)

    @lru_cache(maxsize=1)
    def get_fully_lit_frames(self) -> Set[int]:
        return {frame for frame, index in self.load_light_pattern() if index == 0}

    def load_forground_mask(
        self, frame_id: int, camera_id: str
    ) -> Optional[torch.Tensor]:
        if not self.asset_exists(frame_id):
            return None

        zip_path = self.root_path / "segmentation_parts" / f"cam{camera_id}.zip"
        with zipfile.ZipFile(zip_path, "r") as zipf:
            with zipf.open(f"cam{camera_id}/{frame_id:06d}.png", "r") as png_file:
                img = Image.open(BytesIO(png_file.read()))
                if self.downsample_factor:
                    w, h = img.size
                    img = img.resize(
                        (w // self.downsample_factor, h // self.downsample_factor)
                    )
                fg_mask = torch.from_numpy(
                    (np.array(img) != 0).astype(np.float32)
                ).unsqueeze(0)
                return fg_mask

    def load_image(self, frame: int, camera: str) -> Image:
        zip_path = self.root_path / "image" / f"cam{camera}.zip"

        with zipfile.ZipFile(zip_path, "r") as zipf:
            with zipf.open(f"cam{camera}/{frame:06d}.avif", "r") as avif_file:
                img = Image.open(BytesIO(avif_file.read()))
                if self.downsample_factor:
                    w, h = img.size
                    img = img.resize(
                        (w // self.downsample_factor, h // self.downsample_factor)
                    )
                return to_tensor(img)

    @lru_cache(maxsize=CACHE_LENGTH)
    def load_registration_vertices(self, frame: int) -> Optional[torch.Tensor]:
        zip_path = self.root_path / "kinematic_tracking" / "registration_vertices.zip"
        with zipfile.ZipFile(zip_path, "r") as zipf:
            with zipf.open(f"registration_vertices/{frame:06d}.ply", "r") as ply_file:
                # No faces are included
                pcd = trimesh.exchange.ply.load_ply(
                    ply_file, fix_texture=False, skip_materials=True
                )
                vertices = pcd["vertices"]
                return torch.as_tensor(vertices, dtype=torch.float32) / 1000.0

    @lru_cache(maxsize=1)
    def load_registration_vertices_mean(self) -> np.ndarray:
        mean_path = (
            self.root_path / "kinematic_tracking" / "registration_vertices_mean.npy"
        )
        return np.load(mean_path)

    @lru_cache(maxsize=1)
    def load_registration_vertices_variance(self) -> float:
        verts_path = (
            self.root_path / "kinematic_tracking" / "registration_vertices_variance.txt"
        )
        with open(verts_path, "r") as f:
            return float(f.read())

    @lru_cache(maxsize=1)
    def load_template_mesh(self) -> torch.Tensor:  # Polygon:
        mesh_path = self.root_path / "kinematic_tracking" / "template_mesh.obj"
        with open(mesh_path, "rb") as f:
            mesh = trimesh.load(f)
            vertices = torch.from_numpy(mesh.vertices).float()
            faces = torch.from_numpy(mesh.faces).long()
            return vertices, faces

    @lru_cache(maxsize=1)
    def load_color(self, frame: int) -> Optional[torch.Tensor]:
        if not self.asset_exists(frame):
            # Asset only exists for fully lit frames
            return None

        zip_path = self.root_path / "uv_image" / "color.zip"
        with zipfile.ZipFile(zip_path, "r") as zipf:
            with zipf.open(f"color/{frame:06d}.png", "r") as png_file:
                return to_tensor(Image.open(BytesIO(png_file.read())))

    @lru_cache(maxsize=1)
    def load_color_mean(self) -> torch.Tensor:
        png_path = self.root_path / "uv_image" / "color_mean.png"
        img = Image.open(png_path)
        return to_tensor(img)

    @lru_cache(maxsize=1)
    def load_color_variance(self) -> float:
        color_var_path = self.root_path / "uv_image" / "color_variance.txt"
        with open(color_var_path, "r") as f:
            return float(f.read())

    @lru_cache(maxsize=CACHE_LENGTH)
    def load_head_pose(self, frame_id: int) -> np.ndarray:
        zip_path = self.root_path / "head_pose" / "head_pose.zip"
        with zipfile.ZipFile(zip_path, "r") as zipf:
            with zipf.open(f"{frame_id:06d}.txt", "r") as txt_file:
                lines = txt_file.read().decode("utf-8").splitlines()
                rows = [line.split(" ") for line in lines]
                matrix = np.array(
                    [[float(i) for i in row] for row in rows], dtype=np.float32
                )
                matrix[:3, 3] /= 1000.0  # NOTE: Convert to meters
                return torch.as_tensor(matrix, dtype=torch.float32)

    @lru_cache(maxsize=CACHE_LENGTH)
    def load_background(self, camera: str) -> torch.Tensor:
        zip_path = self.root_path / "per_view_background" / "per_view_background.zip"
        with zipfile.ZipFile(zip_path, "r") as zipf:
            with zipf.open(f"{camera}.png", "r") as png_file:
                return to_tensor(Image.open(BytesIO(png_file.read())))

    @lru_cache(maxsize=1)
    def load_light_pattern(self) -> List[Tuple[int]]:
        light_pattern_path = self.root_path / "lights" / "light_pattern_per_frame.json"
        with open(light_pattern_path, "r") as f:
            return json.load(f)

    @lru_cache(maxsize=1)
    def load_light_pattern_meta(self) -> Dict[str, Any]:
        light_pattern_path = self.root_path / "lights" / "light_pattern_metadata.json"
        with open(light_pattern_path, "r") as f:
            return json.load(f)

    def batch_filter(self, batch):
        batch["image"] = batch["image"].float()
        batch["background"] = batch["background"].float()

        # black level subtraction
        batch["image"][:, 0] -= 2
        batch["image"][:, 1] -= 1
        batch["image"][:, 2] -= 2

        batch["background"][:, 0] -= 2
        batch["background"][:, 1] -= 1
        batch["background"][:, 2] -= 2

        # white balance
        batch["image"][:, 0] *= 1.4
        batch["image"][:, 1] *= 1.1
        batch["image"][:, 2] *= 1.6

        batch["background"][:, 0] *= 1.4
        batch["background"][:, 1] *= 1.1
        batch["background"][:, 2] *= 1.6

        batch["image"] = (batch["image"] / 255.0).clamp(0, 1)
        batch["background"] = (batch["background"] / 255.0).clamp(0, 1)

    @property
    def static_assets(self) -> Dict[str, Any]:
        reg_verts_mean = self.load_registration_vertices_mean()
        reg_verts_var = self.load_registration_vertices_variance()
        light_pattern = self.load_light_pattern()
        light_pattern_meta = self.load_light_pattern_meta()
        color_mean = self.load_color_mean()
        color_var = self.load_color_variance()
        krt = self.load_camera_calibration()

        shared_assets = self.load_shared_assets()

        return {
            "camera_ids": list(krt.keys()),
            "verts_mean": reg_verts_mean,
            "verts_var": reg_verts_var,
            "color_mean": color_mean,
            "color_var": color_var,
            "light_pattern": light_pattern,
            "light_pattern_meta": light_pattern_meta,
            **shared_assets,
        }

    def get(self, frame_id: int, camera_id: str) -> Dict[str, Any]:
        is_fully_lit_frame: bool = frame_id in self.get_fully_lit_frames()

        head_pose = self.load_head_pose(frame_id)
        image = self.load_image(frame_id, camera_id)

        verts = self.load_registration_vertices(frame_id)

        light_pattern = self.load_light_pattern()
        light_pattern = {f[0]: f[1] for f in light_pattern}
        light_pattern_meta = self.load_light_pattern_meta()
        light_pos_all = torch.FloatTensor(light_pattern_meta["light_positions"])
        n_lights_all = light_pos_all.shape[0]
        lightinfo = torch.IntTensor(
            light_pattern_meta["light_patterns"][light_pattern[frame_id]][
                "light_index_durations"
            ]
        )
        n_lights = lightinfo.shape[0]
        light_pos = light_pos_all[lightinfo[:, 0]] / 1000.0
        light_intensity = lightinfo[:, 1:].float() / 5555.0
        light_pos = torch.nn.functional.pad(
            light_pos, (0, 0, 0, n_lights_all - n_lights), "constant", 0
        )
        light_intensity = torch.nn.functional.pad(
            light_intensity, (0, 0, 0, n_lights_all - n_lights), "constant", 0
        )

        fg_mask = self.load_forground_mask(frame_id, camera_id)
        background = self.load_background(camera_id)

        color = self.load_color(frame_id)  # UV image

        if image.size() != background.size():
            background = torch.nn.functional.interpolate(
                background[None], size=(image.shape[1], image.shape[2]), mode="bilinear"
            )[0]

        camera_parameters = self.get_camera_parameters(camera_id)

        # Tracking frame asssets (not available for all frames)
        tracking_frame_assets = {}
        if fg_mask is not None:
            tracking_frame_assets["fg_mask"] = fg_mask
        if color is not None:
            tracking_frame_assets["color"] = color

        return {
            "camera_id": camera_id,
            "frame_id": frame_id,
            "is_fully_lit_frame": is_fully_lit_frame,
            "head_pose": head_pose,
            "image": image,
            "vertices": verts,
            "light_pos": light_pos,
            "light_intensity": light_intensity,
            "n_lights": n_lights,
            "background": background,
            **tracking_frame_assets,
            **camera_parameters,
        }
    
    # def getNerfppNorm(cam_info):
    #     def get_center_and_diag(cam_centers):
    #         cam_centers = np.hstack(cam_centers)
    #         avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
    #         center = avg_cam_center
    #         dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
    #         diagonal = np.max(dist)
    #         return center.flatten(), diagonal

    #     cam_centers = []

    #     for cam in cam_info:
    #         W2C = getWorld2View2(cam.R, cam.T)
    #         C2W = np.linalg.inv(W2C)
    #         cam_centers.append(C2W[:3, 3:4])

    #     center, diagonal = get_center_and_diag(cam_centers)
    #     radius = diagonal * 1.1

    #     translate = -center

    #     return {"translate": translate, "radius": radius}

    @lru_cache(maxsize=1)
    def get_camera_extend(self):
        w2c = torch.stack([torch.tensor(krt["T"]).T for krt in self.load_camera_calibration().values()]) # [N, 4, 4]
        campos = torch.linalg.inv(w2c)[:, :3, 3] / 1_000.0 # [N, 3]
        avg_campos = torch.mean(campos, dim=0, keepdim=True) # [3]
        dist = torch.linalg.norm(campos - avg_campos, dim=1, keepdim=True) # [N]
        diagonal = torch.max(dist)
        radius = diagonal * 1.1
        translate = -avg_campos
        return {"translate": translate, "radius": radius}

    def get_vertices_by_timestep(self, idx: int):
        frame_id = self.frame_list[idx + 250]
        vertices = self.load_registration_vertices(frame_id)
        head_pose = self.load_head_pose(frame_id)
        # print(vertices.min(dim=0).values, vertices.max(dim=0).values)
        vertices = (head_pose[:3, :3] @ vertices.T).T + head_pose[:3, 3]
        # print(vertices.min(dim=0).values, vertices.max(dim=0).values)
        return vertices

    def __len__(self):
        return len(self.frame_list) * len(self.camera_ids)

    def __getitem__(self, idx):
        frame_id = self.frame_list[idx // len(self.camera_ids)]
        camera_id = self.camera_ids[idx % len(self.camera_ids)]

        try:
            sample = self.get(frame_id, camera_id)
            missing_assets = [k for k, v in sample.items() if v is None]
            if len(missing_assets) > 0:
                print(
                    f"Missing assets for frame {frame_id} and camera {camera_id}: {missing_assets}"
                )

                return None

            return sample
        except Exception as e:
            return None


def worker_init_fn(worker_id: int):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)


def collate_fn(items):
    """Modified form of `torch.utils.data.dataloader.default_collate`
    that will strip samples from the batch if they are ``None``."""
    items = [item for item in items if item is not None]
    return default_collate(items) if len(items) > 0 else None


if __name__ == "__main__":
    import argparse
    import cProfile

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=Path, help="Root path to capture data")
    parser.add_argument(
        "-s", "--split", type=str, default="train", choices=["train", "test"]
    )
    args = parser.parse_args()

    root_path = Path(args.input)
    shared_assets_path = root_path.parent / "shared" / "static_assets_head.pt"

    dataset = GoliathHeadDataset(
        root_path=root_path, shared_assets_path=shared_assets_path, split=args.split
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=4,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
    )

    for i, batch in enumerate(iter(dataloader)):
        print(i)
