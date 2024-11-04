import os

os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"

import slangtorch
import torch

slang_pbr = slangtorch.loadModule(os.path.join(os.path.dirname(__file__), "pbr.slang"))


class _pbr_bsdf_slang_func_nhw(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kd, arm, pos, nrm, view_pos, light_pos, light_intensity, min_roughness):
        ctx.save_for_backward(kd, arm, pos, nrm, view_pos, light_pos, light_intensity)
        ctx.min_roughness = min_roughness
        out = slang_pbr.pbr_nhw_fwd(kd, arm, pos, nrm, view_pos, light_pos, light_intensity, min_roughness)
        return out

    @staticmethod
    def backward(ctx, dout):
        kd, arm, pos, nrm, view_pos, light_pos, light_intensity = ctx.saved_tensors
        return slang_pbr.pbr_nhw_bwd(kd, arm, pos, nrm, view_pos, light_pos, light_intensity, ctx.min_roughness, dout.contiguous()) + (None, None, None, None)


# class _pbr_bsdf_slang_func_cn(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, kd, arm, pos, nrm, view_pos, light_pos, min_roughness):
#         ctx.save_for_backward(kd, arm, pos, nrm, view_pos, light_pos)
#         ctx.min_roughness = min_roughness
#         out = slang_pbr.pbr_cn_fwd(kd, arm, pos, nrm, view_pos, light_pos, min_roughness)
#         return out

#     @staticmethod
#     def backward(ctx, dout):
#         kd, arm, pos, nrm, view_pos, light_pos = ctx.saved_tensors
#         return slang_pbr.pbr_cn_bwd(kd, arm, pos, nrm, view_pos, light_pos, ctx.min_roughness, dout.contiguous()) + (
#             None,
#         )


def pbr_bsdf(
    kd: torch.Tensor,
    arm: torch.Tensor,
    pos: torch.Tensor,
    nrm: torch.Tensor,
    view_pos: torch.Tensor,
    light_pos: torch.Tensor,
    light_intensity: torch.Tensor,
    min_roughness: float = 0.08,
):
    """Physically-based bsdf, both diffuse & specular lobes

    All tensors are expected to have a shape of [C, H, W, 3]

    Args:
        kd: Diffuse albedo of shape [C, H, W, 3]
        arm: Armature of shape [C, H, W, 3]
        pos: Surface position of shape [C, H, W, 3]
        nrm: Surface normal of shape [C, H, W, 3]
        view_pos: View position of shape [C, 3]
        light_pos: Light position of shape [C, L, 3]
        min_roughness: Minimum roughness for the specular lobe

    """
    kd = kd.contiguous()
    arm = arm.contiguous()
    pos = pos.contiguous()
    nrm = nrm.contiguous()
    view_pos = view_pos.contiguous()
    light_pos = light_pos.contiguous()
    light_intensity = light_intensity.contiguous()

    if kd.dim() == 4:
        return _pbr_bsdf_slang_func_nhw.apply(kd, arm, pos, nrm, view_pos, light_pos, light_intensity, min_roughness)
    elif kd.dim() == 3:
        # return _pbr_bsdf_slang_func_cn.apply(kd, arm, pos, nrm, view_pos, light_pos, min_roughness)
        raise NotImplementedError("PBR Shading only supports NxHxWxC format")
    else:
        raise ValueError("Invalid input shape")


if __name__ == "__main__":
    from torchvision.utils import save_image

    device = torch.device("cuda:0")

    kd = torch.rand(8, 512, 512, 3, device=device, requires_grad=True)
    arm = torch.rand(8, 512, 512, 3, device=device, requires_grad=True)
    pos = torch.rand(8, 512, 512, 3, device=device, requires_grad=True)
    nrm = torch.rand(8, 512, 512, 3, device=device, requires_grad=True)
    view_pos = torch.rand(8, 3, device=device)
    light_pos = torch.rand(8, 256, 3, device=device) * 10
    light_intensity = torch.rand(8, 256, device=device) * 0.001

    shaded = pbr_bsdf(kd, arm, pos, nrm, view_pos, light_pos, light_intensity)
    loss = shaded.mean()
    loss.backward()

    # batch_size = 1
    # N = 1000

    # kd = torch.rand(batch_size, N, 3, device=device, requires_grad=True)
    # arm = torch.rand(batch_size, N, 3, device=device, requires_grad=True)
    # pos = torch.rand(batch_size, N, 3, device=device, requires_grad=True)
    # nrm = torch.rand(batch_size, N, 3, device=device, requires_grad=True)
    # view_pos = torch.rand(batch_size, 1, 3, device=device, requires_grad=True)
    # light_pos = torch.rand(batch_size, 1, 3, device=device, requires_grad=True)

    # shaded = pbr_bsdf(kd, arm, pos, nrm, view_pos, light_pos)

    # loss = shaded.mean()
    # loss.backward()
    # print(shaded.shape)
