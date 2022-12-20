#include <torch/torch.h>
#include <c10/cuda/CUDAException.h>

#define float1in packed_accessor32<float, 1, torch::RestrictPtrTraits>
#define float2in packed_accessor32<float, 2, torch::RestrictPtrTraits>
using Float2Accessor = torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>;
using Float1Accessor = torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits>;

__global__ static void plane_fitting_kernel(size_t num_voxels,
                                            Float1Accessor xx, Float1Accessor xy, Float1Accessor xz,
                                            Float1Accessor yy, Float1Accessor yz, Float1Accessor zz,
                                            Float2Accessor out) {
    unsigned int voxel_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (voxel_id >= num_voxels) {
        return;
    }
    float xxv = xx[voxel_id], xyv = xy[voxel_id], xzv = xz[voxel_id],
            yyv = yy[voxel_id], yzv = yz[voxel_id], zzv = zz[voxel_id];
    float det_x = yyv * zzv - yzv * yzv;
    float det_y = xxv * zzv - xzv * xzv;
    float det_z = xxv * yyv - xyv * xyv;
    int argmax = 0;
    if (det_x > det_z) {
        argmax = det_x > det_y ? 0 : 1;
    } else {
        argmax = det_z > det_y ? 2 : 1;
    }
    float dx, dy, dz;
    if (argmax == 0) {
        dx = det_x; dy = xzv * yzv - xyv * zzv; dz = xyv * yzv - xzv * yyv;
    } else if (argmax == 1) {
        dx = xzv * yzv - xyv * zzv; dy = det_y; dz = xyv * xzv - yzv * xxv;
    } else {
        dx = xyv * yzv - xzv * yyv; dy = xyv * xzv - yzv * xxv; dz = det_z;
    }
    float n = sqrt(dx * dx + dy * dy + dz * dz) + 1.0e-6;
    out[voxel_id][0] = dx / n; out[voxel_id][1] = dy / n; out[voxel_id][2] = dz / n;
}

torch::Tensor plane_fitting(torch::Tensor xx, torch::Tensor xy, torch::Tensor xz,
                            torch::Tensor yy, torch::Tensor yz, torch::Tensor zz) {
    int num_voxels = xx.size(0);
    torch::Tensor planes = torch::empty({num_voxels, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    {
        dim3 dimBlock = dim3(256);
        dim3 dimGrid = dim3((num_voxels + dimBlock.x - 1) / dimBlock.x);
        plane_fitting_kernel<<<dimGrid, dimBlock>>>(num_voxels,
                                                    xx.float1in(), xy.float1in(), xy.float1in(), yy.float1in(), yz.float1in(), zz.float1in(), planes.float2in());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return planes;
}
