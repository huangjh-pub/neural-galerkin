#include "mc_data.cuh"

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>

#define int1in packed_accessor32<int, 1, torch::RestrictPtrTraits>
#define float1in packed_accessor32<float, 1, torch::RestrictPtrTraits>
#define int2in packed_accessor32<int, 2, torch::RestrictPtrTraits>
#define float2in packed_accessor32<float, 2, torch::RestrictPtrTraits>
#define int3in packed_accessor32<int, 3, torch::RestrictPtrTraits>
#define float3in packed_accessor32<float, 3, torch::RestrictPtrTraits>
using Int1Accessor = torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits>;
using Float1Accessor = torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits>;
using Int2Accessor = torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits>;
using Float2Accessor = torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>;
using Int3Accessor = torch::PackedTensorAccessor32<int, 3, torch::RestrictPtrTraits>;
using Float3Accessor = torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits>;

__device__ static inline float4 sdf_interp(const float3 p1, const float3 p2, float valp1, float valp2) {
    if (fabs(0.0f - valp1) < 1.0e-5f) return make_float4(p1, 1.0);
    if (fabs(0.0f - valp2) < 1.0e-5f) return make_float4(p2, 0.0);
    if (fabs(valp1 - valp2) < 1.0e-5f) return make_float4(p1, 1.0);

    float w2 = (0.0f - valp1) / (valp2 - valp1);
    float w1 = 1 - w2;

    return make_float4(p1.x * w1 + p2.x * w2,
                       p1.y * w1 + p2.y * w2,
                       p1.z * w1 + p2.z * w2, w1);
}

__global__ static void classify_voxels(const Int3Accessor cube_corner_inds,
                                       const Float1Accessor corner_value,
                                       Int1Accessor vertex_counts) {
    const uint num_cubes = cube_corner_inds.size(0);
    const uint cube_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (cube_idx >= num_cubes) {
        return;
    }

    float sdf_vals[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        sdf_vals[i] = corner_value[cube_corner_inds[cube_idx][i][0]];
    }

    // Find triangle config.
    int cube_type = 0;
    if (sdf_vals[0] < 0) cube_type |= 1; if (sdf_vals[1] < 0) cube_type |= 2;
    if (sdf_vals[2] < 0) cube_type |= 4; if (sdf_vals[3] < 0) cube_type |= 8;
    if (sdf_vals[4] < 0) cube_type |= 16; if (sdf_vals[5] < 0) cube_type |= 32;
    if (sdf_vals[6] < 0) cube_type |= 64; if (sdf_vals[7] < 0) cube_type |= 128;

    vertex_counts[cube_idx] = numVertsTable[cube_type];
}

__constant__ int e2iTable[12][2] = {
        {0, 1}, {1, 2}, {2, 3}, {0, 3}, {4, 5}, {5, 6}, {6, 7}, {7, 4}, {0, 4}, {1, 5}, {6, 2}, {3, 7}
};

__global__ static void meshing_cube(const Int3Accessor cube_corner_inds,
                                    const Float1Accessor corner_value,
                                    const Float2Accessor corner_pos,
                                    Float3Accessor triangles,
                                    Int3Accessor vert_ids,
                                    const Int1Accessor count_csum) {
    const uint num_cubes = cube_corner_inds.size(0);
    const uint cube_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (cube_idx >= num_cubes) {
        return;
    }

    float sdf_vals[8];
    int point_ids[8];
    float3 points[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        point_ids[i] = cube_corner_inds[cube_idx][i][0];
        sdf_vals[i] = corner_value[point_ids[i]];
        points[i] = make_float3((float) corner_pos[point_ids[i]][0],
                                (float) corner_pos[point_ids[i]][1],
                                (float) corner_pos[point_ids[i]][2]);
    }

    // Find triangle config.
    int cube_type = 0;
    if (sdf_vals[0] < 0) cube_type |= 1; if (sdf_vals[1] < 0) cube_type |= 2;
    if (sdf_vals[2] < 0) cube_type |= 4; if (sdf_vals[3] < 0) cube_type |= 8;
    if (sdf_vals[4] < 0) cube_type |= 16; if (sdf_vals[5] < 0) cube_type |= 32;
    if (sdf_vals[6] < 0) cube_type |= 64; if (sdf_vals[7] < 0) cube_type |= 128;

    // Find vertex position on each edge (weighted by sdf value)
    int edge_config = edgeTable[cube_type];
    if (edge_config == 0) return;

    float4 vert_list[12];
    if (edge_config & 1) vert_list[0] = sdf_interp(points[0], points[1], sdf_vals[0], sdf_vals[1]);
    if (edge_config & 2) vert_list[1] = sdf_interp(points[1], points[2], sdf_vals[1], sdf_vals[2]);
    if (edge_config & 4) vert_list[2] = sdf_interp(points[2], points[3], sdf_vals[2], sdf_vals[3]);
    if (edge_config & 8) vert_list[3] = sdf_interp(points[3], points[0], sdf_vals[3], sdf_vals[0]);
    if (edge_config & 16) vert_list[4] = sdf_interp(points[4], points[5], sdf_vals[4], sdf_vals[5]);
    if (edge_config & 32) vert_list[5] = sdf_interp(points[5], points[6], sdf_vals[5], sdf_vals[6]);
    if (edge_config & 64) vert_list[6] = sdf_interp(points[6], points[7], sdf_vals[6], sdf_vals[7]);
    if (edge_config & 128) vert_list[7] = sdf_interp(points[7], points[4], sdf_vals[7], sdf_vals[4]);
    if (edge_config & 256) vert_list[8] = sdf_interp(points[0], points[4], sdf_vals[0], sdf_vals[4]);
    if (edge_config & 512) vert_list[9] = sdf_interp(points[1], points[5], sdf_vals[1], sdf_vals[5]);
    if (edge_config & 1024) vert_list[10] = sdf_interp(points[2], points[6], sdf_vals[2], sdf_vals[6]);
    if (edge_config & 2048) vert_list[11] = sdf_interp(points[3], points[7], sdf_vals[3], sdf_vals[7]);

    // Write triangles to array.
    for (int i = 0; triangleTable[cube_type][i] != -1; i += 3) {
        int triangle_id = count_csum[cube_idx] / 3 + i / 3;
#pragma unroll
        for (int vi = 0; vi < 3; ++vi) {
            int vlid = triangleTable[cube_type][i + vi];
            float4 vp_sw = vert_list[vlid];
            triangles[triangle_id][vi][0] = vp_sw.x;
            triangles[triangle_id][vi][1] = vp_sw.y;
            triangles[triangle_id][vi][2] = vp_sw.z;
            int vid0 = point_ids[e2iTable[vlid][0]];
            int vid1 = point_ids[e2iTable[vlid][1]];
            if (vid0 < vid1) {
                int t = vid1;
                vid1 = vid0; vid0 = t;
            }
            vert_ids[triangle_id][vi][0] = vid0;
            vert_ids[triangle_id][vi][1] = vid1;
        }
    }
}

std::vector<torch::Tensor> dual_mc_sparse_cuda(
        const torch::Tensor& cube_corner_inds,     // (M, 8, 2)   int Tensor
        const torch::Tensor& corner_pos,           // (N, 3/4)  float Tensor
        const torch::Tensor& corner_value          // (N, )     float Tensor
) {
    CHECK_INPUT(cube_corner_inds);
    CHECK_INPUT(corner_pos);
    CHECK_INPUT(corner_value);

    const int num_cubes = cube_corner_inds.size(0);
    dim3 dimBlock = dim3(256);
    dim3 dimGrid = dim3((num_cubes + dimBlock.x - 1) / dimBlock.x);

    // Count the number of vertices to be generated.
    torch::Tensor vertex_counts = torch::empty(num_cubes, torch::dtype(torch::kInt32).device(torch::kCUDA));

    if (num_cubes > 0) {
        classify_voxels<<<dimGrid, dimBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
                cube_corner_inds.int3in(),
                corner_value.float1in(),
                vertex_counts.int1in()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // cumsum to determine starting position.
    //    We do not perform compaction, because we are already fast.
    torch::Tensor count_csum = torch::cumsum(vertex_counts.view(-1), 0).to(torch::kInt32);
    int n_triangles = 0;

    if (num_cubes > 0) {
        n_triangles = count_csum[-1].item<int>() / 3;
        count_csum = torch::roll(count_csum, torch::IntList(1));
        count_csum[0] = 0;
        count_csum = count_csum.view(vertex_counts.sizes());
    }

    // Generate triangles
    torch::Tensor triangles = torch::empty({n_triangles, 3, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor vert_ids = torch::empty({n_triangles, 3, 2}, torch::dtype(torch::kInt32).device(torch::kCUDA));

    if (n_triangles > 0) {
        meshing_cube<<<dimGrid, dimBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
                cube_corner_inds.int3in(),
                corner_value.float1in(),
                corner_pos.float2in(),
                triangles.float3in(),
                vert_ids.int3in(),
                count_csum.int1in()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return {triangles, vert_ids};
}
