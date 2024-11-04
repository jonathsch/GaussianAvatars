import os

import slangtorch
import torch

#######################################
# Vector transformation
#######################################

slang_vecmath = slangtorch.loadModule(os.path.join(os.path.dirname(__file__), "points.slang"))


class _xfm_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points: torch.Tensor, matrix: torch.Tensor, is_points: bool):
        ctx.save_for_backward(points, matrix)
        ctx.is_points = is_points
        return slang_vecmath.xfm_fwd(points.contiguous(), matrix, is_points)

    @staticmethod
    def backward(ctx, d_out):
        points, matrix = ctx.saved_variables
        d_points = slang_vecmath.xfm_bwd(points, matrix, d_out, ctx.is_points)
        return d_points, None, None


def transform_vectors(vectors: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
    """
    Batched 3D vector transformation.

    Args:
        vectors: Input vectors of shape [B, H, W, 3] or [B, N, 3]
        mat: Transformation matrix [B, 4, 4]
    """
    B, H, W, _ = vectors.shape
    vectors = vectors.view(B, -1, 3)
    out = torch.matmul(
        torch.nn.functional.pad(vectors, pad=(0, 1), mode="constant", value=0.0),
        torch.transpose(mat, 1, 2),
    )[..., 0:3].contiguous()
    return out.view(B, H, W, 3)


def transform_points(points: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
    """
    Batched 3D point transformation.

    Args:
        points: Input points [B, H, W, 3]
        mat: Transformation matrix [B, 4, 4]
    """
    return _xfm_func.apply(points, mat, True)


#######################################
# Depth map operations
#######################################

slang_depthmap = slangtorch.loadModule(os.path.join(os.path.dirname(__file__), "depth.slang"))


class _unproject_depth_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, depth: torch.Tensor, cam_to_world: torch.Tensor, projection: torch.Tensor):
        ctx.save_for_backward(depth, cam_to_world, projection)
        return slang_depthmap.depth_to_normal_fwd(
            depth.contiguous(), cam_to_world.contiguous(), projection.contiguous()
        )

    @staticmethod
    def backward(ctx, d_out):
        depth, cam_to_world, projection = ctx.saved_variables
        return slang_depthmap.depth_to_normal_bwd(depth, cam_to_world, projection, d_out), None, None


def unproject_depth(depth: torch.Tensor, cam_to_world: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
    """
    Unproject depth map to 3D points.

    Args:
        depth: Depth map [B, H, W, 1]
        cam_to_world: Camera-to-world matrix [B, 4, 4]
        projection: Projection matrix [B, 3, 3]
    """
    B, H, W, _ = depth.shape

    # Normalize projection matrix
    K = torch.eye(3, device=depth.device).unsqueeze(0).repeat(B, 1, 1)  # [B, 3, 3]
    K[:, 0, 0] = projection[:, 0, 0] / W
    K[:, 1, 1] = projection[:, 1, 1] / H
    K[:, 0, 2] = projection[:, 0, 2] / W
    K[:, 1, 2] = projection[:, 1, 2] / H

    # Invert
    inv_K = torch.linalg.inv(K)

    return _unproject_depth_func.apply(depth, cam_to_world, inv_K)
