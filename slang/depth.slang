import utils;

groupshared float3x3 inv_K;
groupshared float4x4 c2w_mtx;

[BackwardDifferentiable]
float3 backproject(const float depth, const no_diff float u, const no_diff float v)
{
    float3 uv_hom = float3(u, v, 1.0f);
    float3 pos_cam = mul(uv_hom, inv_K) * depth;
    float3 pos_world = mul(float4(pos_cam, 1.0f), c2w_mtx).xyz;
    return pos_world;
}

[CudaKernel]
void depth_to_normal_fwd_kernel(TensorView<float> depth, TensorView<float> cam_to_world, TensorView<float> projection, const uint width, const uint height, TensorView<float3> out)
{
    uint3 threadIdx = cudaThreadIdx();
    uint3 blockIdx = cudaBlockIdx();
    uint3 blockDim = cudaBlockDim();

    uint3 idx = blockDim * blockIdx + threadIdx;
    uint3 idxr = idx.zyx;

    float u = divide(idx.x, width);
    float v = divide(idx.y, height);
    uint batch_idx = idx.z;

    if (threadIdx.x < 3 && threadIdx.y < 3)
    {
        inv_K[threadIdx.x][threadIdx.y] = projection[uint3(batch_idx, threadIdx.y, threadIdx.x)];
    }
    if (threadIdx.x < 4 && threadIdx.y < 4)
    {
        c2w_mtx[threadIdx.x][threadIdx.y] = cam_to_world[uint3(batch_idx, threadIdx.y, threadIdx.x)];
    }
    GroupMemoryBarrierWithGroupSync();

    if (idx.x >= out.size(2) || idx.y >= out.size(1) || idx.z >= out.size(0))
    {
        return;
    }

    float3 point = backproject(depth[idxr], u, v);

    out[idxr] = point;
}

[CudaKernel]
void depth_to_normal_bwd_kernel(TensorView<float> depth, TensorView<float> cam_to_world, TensorView<float> projection, const uint width, const uint height, TensorView<float> grad_depth, TensorView<float3> grad_out)
{
    uint3 threadIdx = cudaThreadIdx();
    uint3 blockIdx = cudaBlockIdx();
    uint3 blockDim = cudaBlockDim();

    uint3 idx = blockDim * blockIdx + threadIdx;
    uint3 idxr = idx.zyx;

    float u = divide(idx.x, width);
    float v = divide(idx.y, height);
    uint batch_idx = idx.z;

    if (threadIdx.x < 3 && threadIdx.y < 3)
    {
        inv_K[threadIdx.x][threadIdx.y] = projection[uint3(batch_idx, threadIdx.y, threadIdx.x)];
    }
    if (threadIdx.x < 4 && threadIdx.y < 4)
    {
        c2w_mtx[threadIdx.x][threadIdx.y] = cam_to_world[uint3(batch_idx, threadIdx.y, threadIdx.x)];
    }
    GroupMemoryBarrierWithGroupSync();

    if (idx.x >= grad_out.size(2) || idx.y >= grad_out.size(1) || idx.z >= grad_out.size(0))
    {
        return;
    }

    var dp_depth = diffPair(depth[idxr]);
    var d_out = grad_out[idxr];

    __bwd_diff(backproject)(dp_depth, u, v, d_out);

    grad_depth[idxr] = dp_depth.d;
}

[TorchEntryPoint]
TorchTensor<float3> depth_to_normal_fwd(TorchTensor<float> depth, TorchTensor<float> cam_to_world, TorchTensor<float> projection)
{
    uint3 dims = uint3(depth.size(2), depth.size(1), depth.size(0));
    let blockSize = uint3(8, 8, 1);
    let blockCount = getLaunchGridSize(blockSize, dims);
    var result = TorchTensor<float3>.alloc(dims.z, dims.y, dims.x);

    __dispatch_kernel(depth_to_normal_fwd_kernel, blockCount, blockSize)(depth, cam_to_world, projection, dims.x, dims.y, result);

    return result;
}

[TorchEntryPoint]
TorchTensor<float> depth_to_normal_bwd(TorchTensor<float> depth, TorchTensor<float> cam_to_world, TorchTensor<float> projection, TorchTensor<float3> grad_out)
{
    uint3 dims = uint3(depth.size(2), depth.size(1), depth.size(0));
    let blockSize = uint3(8, 8, 1);
    let blockCount = getLaunchGridSize(blockSize, dims);
    var depth_grad = TorchTensor<float>.emptyLike(depth);

    __dispatch_kernel(depth_to_normal_bwd_kernel, blockCount, blockSize)(depth, cam_to_world, projection, dims.x, dims.y, depth_grad, grad_out);

    return depth_grad;
}
