import utils;

[BackwardDifferentiable]
float lambert(const float3 nrm, const float3 wi)
{
    return max(dot(nrm, wi) / 3.14159265358979323846f, 0.0f);
}

[BackwardDifferentiable]
float3 fresnelSchlick(const float3 f0, const float3 f90, const float cosTheta)
{
    const float SPECULAR_EPSILON = 1e-4f;
    float _cosTheta = clamp(cosTheta, SPECULAR_EPSILON, 1.0f - SPECULAR_EPSILON);
    float scale = pow(1.0f - _cosTheta, 5.0f);
    return f0 * (1.0f - scale) + f90 * scale;
}

[BackwardDifferentiable]
float ndfGGX(const float alphaSqr, const float cosTheta)
{
    const float SPECULAR_EPSILON = 1e-4f;
    float _cosTheta = clamp(cosTheta, SPECULAR_EPSILON, 1.0f - SPECULAR_EPSILON);
    float d = (_cosTheta * alphaSqr - _cosTheta) * _cosTheta + 1.0f;
    return alphaSqr / (d * d * 3.14159265358979323846f);
}

[BackwardDifferentiable]
float lambdaGGX(const float alphaSqr, const float cosTheta)
{
    const float SPECULAR_EPSILON = 1e-4f;
    float _cosTheta = clamp(cosTheta, SPECULAR_EPSILON, 1.0f - SPECULAR_EPSILON);
    float cosThetaSqr = _cosTheta * _cosTheta;
    float tanThetaSqr = (1.0 - cosThetaSqr) / cosThetaSqr;
    return 0.5f * (sqrt(1.0f + alphaSqr * tanThetaSqr) - 1.0f);
}

[BackwardDifferentiable]
float maskingSmithGGXCorrelated(const float alphaSqr, const float cosThetaI, const float cosThetaO)
{
    float lambdaI = lambdaGGX(alphaSqr, cosThetaI);
    float lambdaO = lambdaGGX(alphaSqr, cosThetaO);
    return 1.0f / (1.0f + lambdaI + lambdaO);
}

[BackwardDifferentiable]
float3 pbrSpecular(const float3 col, const float3 nrm, const float3 wo, const float3 wi, const float alpha, no_diff float min_roughness)
{
    float _alpha = clamp(alpha, min_roughness * min_roughness, 1.0f);
    float alphaSqr = _alpha * _alpha;
    const float SPECULAR_EPSILON = 1e-4f;
    float woDotN = dot(wo, nrm);
    float wiDotN = dot(wi, nrm);

    bool frontfacing = (woDotN > SPECULAR_EPSILON) & (wiDotN > SPECULAR_EPSILON);

    float3 res = float3(0.0f);
    if (frontfacing)
    {
        float3 h = safeNormalize(wo + wi);
        float woDotH = dot(wo, h);
        float nDotH = dot(nrm, h);
        float D = ndfGGX(alphaSqr, nDotH);
        float G = maskingSmithGGXCorrelated(alphaSqr, woDotN, wiDotN);
        float3 F = fresnelSchlick(col, 1.0f, woDotH);
        res = F * D * G * 0.25 / woDotN;
    }
    return res;
}

//////////////////////////////////////////////////////////////////////////
// Full PBR BSDF
//////////////////////////////////////////////////////////////////////////
[BackwardDifferentiable]
float3 pbrBSDF(const float3 kd, const float3 arm, const float3 pos, const float3 nrm, no_diff const float3 view_pos,
               no_diff const float3 light_pos, no_diff const float light_intensity, no_diff float min_roughness)
{
    float3 wo = safeNormalize(view_pos - pos);
    float3 wi = safeNormalize(light_pos - pos);

    float alpha = arm.y * arm.y;
    float3 spec_col = (0.04f * (1.0f - arm.z) + kd * arm.z) * (1.0 - arm.x);
    float3 diff_col = kd * (1.0f - arm.z);

    float diff = lambert(nrm, wi);
    float3 diffuse = diff_col * diff;
    float3 specular = pbrSpecular(spec_col, nrm, wo, wi, alpha, min_roughness);

    let relativeLightIntensity = light_intensity / length(pos - light_pos);

    return (diffuse + specular) * relativeLightIntensity;
}

/////////////////////////////////////////////////////////////////////////////////
// Cuda kernels
/////////////////////////////////////////////////////////////////////////////////
[CudaKernel]
void pbr_nhw_fwd_kernel(TensorView<float3> kd,             // B x H x W x 3
                        TensorView<float3> arm,            // B x H x W x 3
                        TensorView<float3> pos,            // B x H x W x 3
                        TensorView<float3> nrm,            // B x H x W x 3
                        TensorView<float3> view_pos,       // B x 3
                        TensorView<float3> light_pos,      // B x L x 3
                        TensorView<float> light_intensity, // B x L x 1
                        const float min_roughness,
                        TensorView<float3> output) // B x H x W x 3
{
    uint3 idx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
    if (idx.x > output.size(2) || idx.y > output.size(1), idx.z > output.size(0))
    {
        return;
    }

    uint3 idxr = uint3(idx.z, idx.y, idx.x);
    float3 v_pos = view_pos[idx.z];

    let n_lights = light_pos.size(1);
    output[idxr] = float3(0.0f);
    for (uint i = 0; i < n_lights; ++i)
    {
        uint2 l_idx = uint2(idx.z, i);
        float3 l_pos = light_pos[l_idx];
        float l_int = light_intensity[l_idx];
        float3 res = pbrBSDF(kd[idxr], arm[idxr], pos[idxr], nrm[idxr], v_pos, l_pos, l_int, min_roughness);
        output[idxr] += res;
    }
    // float3 l_pos = broadcast_fetch(light_pos, idxr);

    // float3 res = pbrBSDF(kd[idxr], arm[idxr], pos[idxr], nrm[idxr], v_pos, l_pos, min_roughness);

    // output[idxr] = res;
}

[CudaKernel]
void pbr_nhw_bwd_kernel(TensorView<float3> kd,
                        TensorView<float3> arm,
                        TensorView<float3> pos,
                        TensorView<float3> nrm,
                        TensorView<float3> view_pos,
                        TensorView<float3> light_pos,
                        TensorView<float> light_intensity,
                        const float min_roughness,
                        TensorView<float3> kd_grad,
                        TensorView<float3> arm_grad,
                        TensorView<float3> pos_grad,
                        TensorView<float3> nrm_grad,
                        // TensorView<float3> view_pos_grad,
                        // TensorView<float3> light_pos_grad,
                        TensorView<float3> grad_out)
{
    uint3 idx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
    if (idx.x > grad_out.size(2) || idx.y > grad_out.size(1) || idx.z > grad_out.size(0))
    {
        return;
    }

    uint3 idxr = uint3(idx.z, idx.y, idx.x);

    float3 v_pos = view_pos[idx.z];

    let n_lights = light_pos.size(1);
    kd_grad[idxr] = float3(0.0f);
    arm_grad[idxr] = float3(0.0f);
    pos_grad[idxr] = float3(0.0f);
    nrm_grad[idxr] = float3(0.0f);
    for (uint i = 0; i < n_lights; ++i)
    {
        uint2 l_idx = uint2(idx.z, i);
        float3 l_pos = light_pos[l_idx];
        float l_int = light_intensity[l_idx];

        float3 dout = grad_out[idxr];
        var dp_kd = diffPair(kd[idxr]);
        var dp_arm = diffPair(arm[idxr]);
        var dp_pos = diffPair(pos[idxr]);
        var dp_nrm = diffPair(nrm[idxr]);

        __bwd_diff(pbrBSDF)(dp_kd, dp_arm, dp_pos, dp_nrm, v_pos, l_pos, l_int,
                            min_roughness, dout);

        kd_grad[idxr] += dp_kd.d;
        arm_grad[idxr] += dp_arm.d;
        pos_grad[idxr] += dp_pos.d;
        nrm_grad[idxr] += dp_nrm.d;
    }

    // float3 l_pos = broadcast_fetch(light_pos, idxr);

    // var dp_kd = diffPair(kd[idxr]);
    // var dp_arm = diffPair(arm[idxr]);
    // var dp_pos = diffPair(pos[idxr]);
    // var dp_nrm = diffPair(nrm[idxr]);
    // var dp_view_pos = diffPair(v_pos);
    // var dp_light_pos = diffPair(l_pos);
    // var dout = grad_out[idxr];

    // __bwd_diff(pbrBSDF)(dp_kd, dp_arm, dp_pos, dp_nrm, dp_view_pos, dp_light_pos,
    //                     min_roughness, dout);

    // kd_grad[idxr] = dp_kd.d;
    // arm_grad[idxr] = dp_arm.d;
    // pos_grad[idxr] = dp_pos.d;
    // nrm_grad[idxr] = dp_nrm.d;
    // broadcast_store(view_pos_grad, idxr, dp_view_pos.d);
    // broadcast_store(light_pos_grad, idxr, dp_light_pos.d);
}

// [CudaKernel]
// void pbr_cn_fwd_kernel(TensorView<float3> kd,
//                        TensorView<float3> arm,
//                        TensorView<float3> pos,
//                        TensorView<float3> nrm,
//                        TensorView<float3> view_pos,
//                        TensorView<float3> light_pos,
//                        const float min_roughness,
//                        TensorView<float3> output)
// {
//     uint3 idx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
//     if (idx.x > output.size(1) || idx.y > output.size(0))
//     {
//         return;
//     }

//     uint2 idxr = uint2(idx.y, idx.x);

//     float3 v_pos = broadcast_fetch(view_pos, idxr);
//     float3 l_pos = broadcast_fetch(light_pos, idxr);

//     float3 res = pbrBSDF(kd[idxr], arm[idxr], pos[idxr], nrm[idxr], v_pos, l_pos, min_roughness);

//     output[idxr] = res;
// }

// [CudaKernel]
// void pbr_cn_bwd_kernel(TensorView<float3> kd,
//                        TensorView<float3> arm,
//                        TensorView<float3> pos,
//                        TensorView<float3> nrm,
//                        TensorView<float3> view_pos,
//                        TensorView<float3> light_pos,
//                        const float min_roughness,
//                        TensorView<float3> kd_grad,
//                        TensorView<float3> arm_grad,
//                        TensorView<float3> pos_grad,
//                        TensorView<float3> nrm_grad,
//                        TensorView<float3> view_pos_grad,
//                        TensorView<float3> light_pos_grad,
//                        TensorView<float3> grad_out)
// {
//     uint3 idx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
//     if (idx.x > grad_out.size(1) || idx.y > grad_out.size(0))
//     {
//         return;
//     }

//     uint2 idxr = idx.yx;

//     float3 v_pos = broadcast_fetch(view_pos, idxr);
//     float3 l_pos = broadcast_fetch(light_pos, idxr);

//     var dp_kd = diffPair(kd[idxr]);
//     var dp_arm = diffPair(arm[idxr]);
//     var dp_pos = diffPair(pos[idxr]);
//     var dp_nrm = diffPair(nrm[idxr]);
//     var dp_view_pos = diffPair(v_pos);
//     var dp_light_pos = diffPair(l_pos);
//     var dout = grad_out[idxr];

//     __bwd_diff(pbrBSDF)(dp_kd, dp_arm, dp_pos, dp_nrm, dp_view_pos, dp_light_pos,
//                         min_roughness, dout);

//     kd_grad[idxr] = dp_kd.d;
//     arm_grad[idxr] = dp_arm.d;
//     pos_grad[idxr] = dp_pos.d;
//     nrm_grad[idxr] = dp_nrm.d;
//     broadcast_store(view_pos_grad, idxr, dp_view_pos.d);
//     broadcast_store(light_pos_grad, idxr, dp_light_pos.d);
// }

/////////////////////////////////////////////////////////////////////////////////
// Torch entry points
/////////////////////////////////////////////////////////////////////////////////
// [TorchEntryPoint]
// TorchTensor<float3> pbr_cn_fwd(TorchTensor<float3> kd,
//                                TorchTensor<float3> arm,
//                                TorchTensor<float3> pos,
//                                TorchTensor<float3> nrm,
//                                TorchTensor<float3> view_pos,
//                                TorchTensor<float3> light_pos,
//                                const float min_roughness)
// {
//     // C x N x 3
//     var result = TorchTensor<float3>.emptyLike(kd);
//     uint3 dims = uint3(result.size(1), result.size(0), 1);
//     let blockSize = uint3(8 * 8, 1, 1);
//     let blockCount = getLaunchGridSize(blockSize, dims);

//     __dispatch_kernel(pbr_cn_fwd_kernel, blockCount, blockSize)(kd, arm, pos, nrm, view_pos, light_pos, min_roughness, result);

//     return result;
// }

[TorchEntryPoint]
TorchTensor<float3> pbr_nhw_fwd(TorchTensor<float3> kd,
                                TorchTensor<float3> arm,
                                TorchTensor<float3> pos,
                                TorchTensor<float3> nrm,
                                TorchTensor<float3> view_pos,
                                TorchTensor<float3> light_pos,
                                TorchTensor<float> light_intensity,
                                const float min_roughness)
{
    // C x H x W x 3
    var result = TorchTensor<float3>.emptyLike(kd);
    uint3 dims = uint3(result.size(2), result.size(1), result.size(0));
    let blockSize = uint3(8, 8, 1);
    let blockCount = getLaunchGridSize(blockSize, dims);

    __dispatch_kernel(pbr_nhw_fwd_kernel, blockCount, blockSize)(kd, arm, pos, nrm, view_pos, light_pos, light_intensity, min_roughness, result);

    return result;
}

struct PbrBwdReturnType
{
    TorchTensor<float3> kd_grad;
    TorchTensor<float3> arm_grad;
    TorchTensor<float3> pos_grad;
    TorchTensor<float3> nrm_grad;
    TorchTensor<float3> view_pos_grad;
    TorchTensor<float3> light_pos_grad;
}

// [TorchEntryPoint]
// PbrBwdReturnType pbr_cn_bwd(TorchTensor<float3> kd,
//                             TorchTensor<float3> arm,
//                             TorchTensor<float3> pos,
//                             TorchTensor<float3> nrm,
//                             TorchTensor<float3> view_pos,
//                             TorchTensor<float3> light_pos,
//                             const float min_roughness,
//                             TorchTensor<float3> grad_out)
// {
//     PbrBwdReturnType result;

//     result.kd_grad = TorchTensor<float3>.emptyLike(kd);
//     result.arm_grad = TorchTensor<float3>.emptyLike(arm);
//     result.pos_grad = TorchTensor<float3>.emptyLike(pos);
//     result.nrm_grad = TorchTensor<float3>.emptyLike(nrm);
//     result.view_pos_grad = TorchTensor<float3>.emptyLike(view_pos);
//     result.light_pos_grad = TorchTensor<float3>.emptyLike(light_pos);

//     uint3 dims = uint3(grad_out.size(1), grad_out.size(0), 1);
//     let blockSize = uint3(8 * 8, 1, 1);
//     let blockCount = getLaunchGridSize(blockSize, dims);

//     __dispatch_kernel(pbr_cn_bwd_kernel, blockCount, blockSize)(kd, arm, pos, nrm, view_pos, light_pos, min_roughness,
//                                                                 result.kd_grad, result.arm_grad, result.pos_grad, result.nrm_grad, result.view_pos_grad, result.light_pos_grad, grad_out);

//     return result;
// }

[TorchEntryPoint]
PbrBwdReturnType pbr_nhw_bwd(TorchTensor<float3> kd,
                             TorchTensor<float3> arm,
                             TorchTensor<float3> pos,
                             TorchTensor<float3> nrm,
                             TorchTensor<float3> view_pos,
                             TorchTensor<float3> light_pos,
                             TorchTensor<float> light_intensity,
                             const float min_roughness,
                             TorchTensor<float3> grad_out)
{
    PbrBwdReturnType result;

    result.kd_grad = TorchTensor<float3>.emptyLike(kd);
    result.arm_grad = TorchTensor<float3>.emptyLike(arm);
    result.pos_grad = TorchTensor<float3>.emptyLike(pos);
    result.nrm_grad = TorchTensor<float3>.emptyLike(nrm);
    result.view_pos_grad = TorchTensor<float3>.emptyLike(view_pos);
    result.light_pos_grad = TorchTensor<float3>.emptyLike(light_pos);

    uint3 dims = uint3(grad_out.size(2), grad_out.size(1), grad_out.size(0));
    let blockSize = uint3(8, 8, 1);
    let blockCount = getLaunchGridSize(blockSize, dims);

    __dispatch_kernel(pbr_nhw_bwd_kernel, blockCount, blockSize)(kd, arm, pos, nrm, view_pos, light_pos, light_intensity, min_roughness,
                                                                 result.kd_grad, result.arm_grad, result.pos_grad, result.nrm_grad, grad_out);

    return result;
}
