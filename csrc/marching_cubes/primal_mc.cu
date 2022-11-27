#include "mc_data.cuh"

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>

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

__global__ static void classify_voxels(const CubeBaseAccessor cube_base,
                                       const CubeSDFAccessor cube_sdf,
                                       VoxelTypeAccessor vertex_counts) {
    const uint r = cube_sdf.size(1) - 1;
    const uint r3 = r * r * r;
    const uint num_lif = cube_base.size(0);
    const float sbs = 1.0f / r;         // sub-block-size

    const uint lif_id = blockIdx.x * blockDim.x + threadIdx.x;
    const uint sub_id = blockIdx.y * blockDim.y + threadIdx.y;

    if (lif_id >= num_lif || sub_id >= r3) {
        return;
    }

    const uint rx = sub_id / (r * r);
    const uint ry = (sub_id / r) % r;
    const uint rz = sub_id % r;

    float bx = cube_base[lif_id][0];
    float by = cube_base[lif_id][1];
    float bz = cube_base[lif_id][2];

    float sdf_vals[8];
    sdf_vals[0] = cube_sdf[lif_id][rx][ry][rz];
    sdf_vals[1] = cube_sdf[lif_id][rx + 1][ry][rz];
    sdf_vals[2] = cube_sdf[lif_id][rx + 1][ry + 1][rz];
    sdf_vals[3] = cube_sdf[lif_id][rx][ry + 1][rz];
    sdf_vals[4] = cube_sdf[lif_id][rx][ry][rz + 1];
    sdf_vals[5] = cube_sdf[lif_id][rx + 1][ry][rz + 1];
    sdf_vals[6] = cube_sdf[lif_id][rx + 1][ry + 1][rz + 1];
    sdf_vals[7] = cube_sdf[lif_id][rx][ry + 1][rz + 1];

    // Find triangle config.
    int cube_type = 0;
    if (sdf_vals[0] < 0) cube_type |= 1; if (sdf_vals[1] < 0) cube_type |= 2;
    if (sdf_vals[2] < 0) cube_type |= 4; if (sdf_vals[3] < 0) cube_type |= 8;
    if (sdf_vals[4] < 0) cube_type |= 16; if (sdf_vals[5] < 0) cube_type |= 32;
    if (sdf_vals[6] < 0) cube_type |= 64; if (sdf_vals[7] < 0) cube_type |= 128;

    vertex_counts[lif_id][rx][ry][rz] = numVertsTable[cube_type];
}

__global__ static void meshing_cube(const CubeBaseAccessor cube_base,
                                    const CubeSDFAccessor cube_sdf,
                                    TrianglesAccessor triangles,
                                    VertIDAccessor vert_ids,
                                    VertIDAccessor src_ids,
                                    TrianglesAccessor src_weight,
                                    const VoxelTypeAccessor count_csum) {
    const uint r = cube_sdf.size(1) - 1;
    const uint r3 = r * r * r;
    const uint num_lif = cube_base.size(0);
    const float sbs = 1.0f / r;         // sub-block-size

    const uint lif_id = blockIdx.x * blockDim.x + threadIdx.x;
    const uint sub_id = blockIdx.y * blockDim.y + threadIdx.y;

    if (lif_id >= num_lif || sub_id >= r3) {
        return;
    }

    const uint rx = sub_id / (r * r);
    const uint ry = (sub_id / r) % r;
    const uint rz = sub_id % r;

    float bx = cube_base[lif_id][0];
    float by = cube_base[lif_id][1];
    float bz = cube_base[lif_id][2];

    const int sid0 = (r + 1) * (r + 1) * (r + 1);
    const int sid1 = (r + 1) * (r + 1);
    const int sid2 = (r + 1);

    // Find all 8 neighbours
    float3 points[8];
    float sdf_vals[8];

    sdf_vals[0] = cube_sdf[lif_id][rx][ry][rz];
    points[0] = make_float3(bx + rx * sbs, by + ry * sbs, bz + rz * sbs);

    sdf_vals[1] = cube_sdf[lif_id][rx + 1][ry][rz];
    points[1] = make_float3(bx + (rx + 1) * sbs, by + ry * sbs, bz + rz * sbs);

    sdf_vals[2] = cube_sdf[lif_id][rx + 1][ry + 1][rz];
    points[2] = make_float3(bx + (rx + 1) * sbs, by + (ry + 1) * sbs, bz + rz * sbs);

    sdf_vals[3] = cube_sdf[lif_id][rx][ry + 1][rz];
    points[3] = make_float3(bx + rx * sbs, by + (ry + 1) * sbs, bz + rz * sbs);

    sdf_vals[4] = cube_sdf[lif_id][rx][ry][rz + 1];
    points[4] = make_float3(bx + rx * sbs, by + ry * sbs, bz + (rz + 1) * sbs);

    sdf_vals[5] = cube_sdf[lif_id][rx + 1][ry][rz + 1];
    points[5] = make_float3(bx + (rx + 1) * sbs, by + ry * sbs, bz + (rz + 1) * sbs);

    sdf_vals[6] = cube_sdf[lif_id][rx + 1][ry + 1][rz + 1];
    points[6] = make_float3(bx + (rx + 1) * sbs, by + (ry + 1) * sbs, bz + (rz + 1) * sbs);

    sdf_vals[7] = cube_sdf[lif_id][rx][ry + 1][rz + 1];
    points[7] = make_float3(bx + rx * sbs, by + (ry + 1) * sbs, bz + (rz + 1) * sbs);

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
        int triangle_id = count_csum[lif_id][rx][ry][rz] / 3 + i / 3;
#pragma unroll
        for (int vi = 0; vi < 3; ++vi) {
            int vlid = triangleTable[cube_type][i + vi];
            float4 vp_sw = vert_list[vlid];
            triangles[triangle_id][vi][0] = vp_sw.x;
            triangles[triangle_id][vi][1] = vp_sw.y;
            triangles[triangle_id][vi][2] = vp_sw.z;
            vert_ids[triangle_id][vi][0] = int(bx) * r + rx + idTable[vlid][0];
            vert_ids[triangle_id][vi][1] = int(by) * r + ry + idTable[vlid][1];
            vert_ids[triangle_id][vi][2] = int(bz) * r + rz + idTable[vlid][2];
            vert_ids[triangle_id][vi][3] = idTable[vlid][3];
            src_ids[triangle_id][vi][0] = sid0 * lif_id +
                    sid1 * (rx + linkTable[vlid][0]) + sid2 * (ry + linkTable[vlid][1]) + (rz + linkTable[vlid][2]);
            src_ids[triangle_id][vi][1] = sid0 * lif_id +
                    sid1 * (rx + linkTable[vlid][3]) + sid2 * (ry + linkTable[vlid][4]) + (rz + linkTable[vlid][5]);
            src_weight[triangle_id][vi][0] = vp_sw.w;
        }
    }
}

std::vector<torch::Tensor> marching_cubes_sparse_cuda(
        torch::Tensor cube_base,            // (M, 3)
        torch::Tensor cube_sdf             // (M, rx, ry, rz)
) {
    CHECK_INPUT(cube_base);
    CHECK_INPUT(cube_sdf);

    const int r = cube_sdf.size(1) - 1;
    const int r3 = r * r * r;
    const int num_lif = cube_base.size(0);

    dim3 dimBlock = dim3(16, 16);
    uint xBlocks = (num_lif + dimBlock.x - 1) / dimBlock.x;
    uint yBlocks = (r3 + dimBlock.y - 1) / dimBlock.y;
    dim3 dimGrid = dim3(xBlocks, yBlocks);

    // Count the number of vertices to be generated.
    //      As the entire MC is really fast (10k times faster than linear solver), no need to further tune it.
    //      More improvement can be obtained by using shared mem within the kernel:
    //  See https://github.com/NVIDIA/cuda-samples/blob/master/Samples/marchingCubes/marchingCubes_kernel.cu
    torch::Tensor vertex_counts = torch::empty({num_lif, r, r, r}, torch::dtype(torch::kInt32).device(torch::kCUDA));

    if (num_lif > 0) {
        classify_voxels<<<dimGrid, dimBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
                cube_base.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                cube_sdf.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                vertex_counts.packed_accessor32<int, 4, torch::RestrictPtrTraits>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // cumsum to determine starting position.
    //    We do not perform compaction, because we are already fast.
    torch::Tensor count_csum = torch::cumsum(vertex_counts.view(-1), 0).to(torch::kInt32);
    int n_triangles = 0;

    if (num_lif > 0) {
        n_triangles = count_csum[-1].item<int>() / 3;
        count_csum = torch::roll(count_csum, torch::IntList(1));
        count_csum[0] = 0;
        count_csum = count_csum.view(vertex_counts.sizes());
    }

    // Generate triangles
    torch::Tensor triangles = torch::empty({n_triangles, 3, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor vert_ids = torch::empty({n_triangles, 3, 4}, torch::dtype(torch::kInt32).device(torch::kCUDA));

    // for back-propagation.
    torch::Tensor src_ids = torch::empty({n_triangles, 3, 2}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    torch::Tensor src_weight = torch::empty({n_triangles, 3, 1}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    if (n_triangles > 0) {
        meshing_cube<<<dimGrid, dimBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
                cube_base.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                cube_sdf.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                triangles.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                vert_ids.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
                src_ids.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
                src_weight.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                count_csum.packed_accessor32<int, 4, torch::RestrictPtrTraits>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return {triangles, vert_ids, src_ids, src_weight};
}
