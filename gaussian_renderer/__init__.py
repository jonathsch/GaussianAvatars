#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import Optional

import torch
import math
from typing import Union
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from scene import GaussianModel, FlameGaussianModel
from utils.sh_utils import eval_sh
import gsplat


def render(
    viewpoint_camera,
    pc: Union[GaussianModel, FlameGaussianModel],
    pipe,
    bg_color: torch.Tensor,
    bg_img: Optional[torch.Tensor] = None,
    scaling_modifier=1.0,
    override_color=None,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    # try:
    #     screenspace_points.retain_grad()
    # except:
    #     pass

    # Set up rasterization configuration
    # tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    # tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # raster_settings = GaussianRasterizationSettings(
    #     image_height=int(viewpoint_camera.image_height),
    #     image_width=int(viewpoint_camera.image_width),
    #     tanfovx=tanfovx,
    #     tanfovy=tanfovy,
    #     bg=bg_color,
    #     scale_modifier=scaling_modifier,
    #     viewmatrix=viewpoint_camera.world_view_transform.cuda(),
    #     projmatrix=viewpoint_camera.full_proj_transform.cuda(),
    #     sh_degree=pc.active_sh_degree,
    #     campos=viewpoint_camera.camera_center.cuda(),
    #     prefiltered=False,
    #     debug=pipe.debug,
    # )

    # rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    # means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree + 1) ** 2
            )
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
                pc.get_features.shape[0], 1
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    # rendered_image, radii = rasterizer(
    #     means3D=means3D,
    #     means2D=means2D,
    #     shs=shs,
    #     colors_precomp=colors_precomp,
    #     opacities=opacity,
    #     scales=scales,
    #     rotations=rotations,
    #     cov3D_precomp=cov3D_precomp,
    # )

    # rendered_image, alpha, meta = gsplat.rasterization(
    #     means=means3D,
    #     quats=rotations,
    #     scales=scales,
    #     opacities=opacity.squeeze(-1),
    #     colors=colors_precomp if colors_precomp is not None else shs,
    #     viewmats=viewpoint_camera.world_view_transform.cuda()
    #     .transpose(0, 1)
    #     .unsqueeze(0),
    #     Ks=viewpoint_camera.K.cuda().unsqueeze(0),
    #     width=viewpoint_camera.image_width,
    #     height=viewpoint_camera.image_height,
    #     sh_degree=pc.active_sh_degree,
    # )

    pbr_materials = pc.get_pbr_material

    (
        rendered_image,
        alpha,
        render_normals,
        normals_from_depth,
        render_dist,
        render_median,
        meta,
    ) = gsplat.rasterization_2dgs(
        means=means3D,
        quats=rotations,
        scales=scales,
        opacities=opacity.squeeze(-1),
        colors=colors_precomp if colors_precomp is not None else shs,
        viewmats=viewpoint_camera.world_view_transform.cuda()
        .transpose(0, 1)
        .unsqueeze(0),
        Ks=viewpoint_camera.K.cuda().unsqueeze(0),
        width=viewpoint_camera.image_width,
        height=viewpoint_camera.image_height,
        sh_degree=pc.active_sh_degree,
        render_mode="RGB+D",
    )
    rendered_image = rendered_image[..., :3].contiguous()

    # with torch.no_grad():
    #     g_buffer, _, _, _, _, _,  _ = gsplat.rasterization_2dgs(
    #         means=means3D,
    #         quats=rotations,
    #         scales=scales,
    #         opacities=opacity.squeeze(-1),
    #         colors=pbr_materials,
    #         viewmats=viewpoint_camera.world_view_transform.cuda()
    #         .transpose(0, 1)
    #         .unsqueeze(0),
    #         Ks=viewpoint_camera.K.cuda().unsqueeze(0),
    #         width=viewpoint_camera.image_width,
    #         height=viewpoint_camera.image_height,
    #         # sh_degree=pc.active_sh_degree,
    #         render_mode="RGB"
    #     )
    #     kd = g_buffer[..., :3].squeeze(0).permute(2, 0, 1).clamp(0, 1) # [3, H, W]
    #     ks = g_buffer[..., 3:6].squeeze(0).permute(2, 0, 1).clamp(0, 1) # [3, H, W]

    rendered_image = rendered_image.squeeze(0).permute(2, 0, 1)  # [3, H, W]
    alpha = alpha.squeeze(0).permute(2, 0, 1)  # [1, H, W]

    if bg_img is not None:
        rendered_image = rendered_image * alpha + bg_img * (1 - alpha)

    meta["gradient_2dgs"].retain_grad()
    radii = torch.zeros(
        means3D.shape[0], device=meta["radii"].device, dtype=meta["radii"].dtype
    )
    radii[meta["gaussian_ids"]] = meta["radii"]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.

    # return {
    #     "render": rendered_image,
    #     "viewspace_points": screenspace_points,
    #     "visibility_filter": radii > 0,
    #     "radii": radii,
    #     "gaussian_ids": None,
    #     # "alpha": alpha,
    # }

    return {
        "render": rendered_image,
        "viewspace_points": meta["gradient_2dgs"],
        "visibility_filter": radii > 0,
        "radii": radii,
        "meta": meta,
        "alpha": alpha,
        "render_normals": render_normals.squeeze(0).permute(2, 0, 1),  # [3, H, W]
        "normals_from_depth": normals_from_depth.squeeze(0).permute(
            2, 0, 1
        ),  # [3, H, W]
    }


def render_gsplat(
    world_to_cam: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    pc: Union[GaussianModel, FlameGaussianModel],
):
    means3d = pc.get_xyz
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    shs = pc.get_features
    
    (
        rendered_image,
        alpha,
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
        colors=shs,
        viewmats=world_to_cam.unsqueeze(0),
        Ks=K.unsqueeze(0),
        width=width,
        height=height,
        sh_degree=pc.active_sh_degree,
        render_mode="RGB+ED",
    )
    rendered_image, render_depth = rendered_image[..., :3].contiguous(), rendered_image[..., 3:4].contiguous()

    rendered_image = rendered_image.squeeze(0) # [H, W, 3]
    alpha = alpha.squeeze(0).repeat(1, 1, 3) # [H, W, 3]
    render_depth = render_depth.squeeze(0).repeat(1, 1, 3) # [H, W, 3]

    render_normals = render_normals.squeeze(0) * 0.5 + 0.5 # [H, W, 3]
    normals_from_depth = normals_from_depth.squeeze(0) * 0.5 + 0.5 # [H, W, 3]

    return {
        "render": rendered_image,
        "alpha": alpha,
        "depth": render_depth,
        "render_normals": render_normals,
        "normals_from_depth": normals_from_depth
    }
