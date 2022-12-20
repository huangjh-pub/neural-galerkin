#include <torch/torch.h>
#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <pybind11/pytypes.h>
#include "../common/hashmap_cuda.cuh"

namespace py = pybind11;

#define int2in packed_accessor32<int, 2, torch::RestrictPtrTraits>
#define int3in packed_accessor32<int, 3, torch::RestrictPtrTraits>
using Int2Accessor = torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits>;
using Int3Accessor = torch::PackedTensorAccessor32<int, 3, torch::RestrictPtrTraits>;

// From Face ID to its axis (fix_axis_small/big, fix_axis, iter_axis0, iter_axis1) and MC accumulate indices.
__constant__ int faceAxisTable[6][4] = {
        {0, 0, 1, 2}, {1, 0, 1, 2}, {0, 1, 2, 0}, {1, 1, 2, 0}, {0, 2, 0, 1}, {1, 2, 0, 1}
};
__constant__ int faceAccIndsTable[6][4] = {
        {1, 2, 5, 6}, {0, 3, 4, 7}, {2, 3, 6, 7}, {0, 1, 5, 4}, {4, 5, 6, 7}, {0, 1, 2, 3}
};

// From Edge ID to its axis and MC accumulate indices.
__constant__ int edgeAxisTable[12][5] = {
        {0, 0, 1, 2, 0}, {0, 1, 1, 2, 0}, {1, 0, 1, 2, 0}, {1, 1, 1, 2, 0},
        {0, 0, 2, 0, 1}, {0, 1, 2, 0, 1}, {1, 0, 2, 0, 1}, {1, 1, 2, 0, 1},
        {0, 0, 0, 1, 2}, {0, 1, 0, 1, 2}, {1, 0, 0, 1, 2}, {1, 1, 0, 1, 2},
};
__constant__ int edgeAccIndsTable[12][2] = {
        {6, 7}, {2, 3}, {4, 5}, {0, 1},
        {5, 6}, {4, 7}, {1, 2}, {0, 3},
        {2, 6}, {1, 5}, {3, 7}, {0, 4}
};

// From Corner ID to corresponding values
__constant__ int cornerAxisTable[8][3] = {
        {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1},
        {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}
};
__constant__ int cornerAccIndsTable[8] = {
        6, 2, 5, 1, 7, 3, 4, 0
};


__global__ static void acc_inds_kernel(size_t n_primals, int stride, int d, int idx_base,
                                       HashLookupParam param,
                                       const Int2Accessor primal_base,
                                       Int3Accessor acc_inds,
                                       int* acc_inds_cnt) {
    int corner_id = blockIdx.x * blockDim.x + threadIdx.x;

    int cube_corner_count = 8;
    int cube_edge_count = (stride - 1) * 12;
    int cube_face_count = (stride - 1) * (stride - 1) * 6;

    int cube_total_count = cube_corner_count + cube_edge_count + cube_face_count;
    int primal_id = corner_id / cube_total_count;
    int cube_id = corner_id % cube_total_count;
    if (primal_id >= n_primals) {
        return;
    }

    if (cube_id >= cube_corner_count + cube_edge_count) {
        // Face attached, should accumulate 4 MC inds if DC exists.
        cube_id -= (cube_corner_count + cube_edge_count);
        int face_count = (stride - 1) * (stride - 1);
        int face_id = cube_id / face_count;
        int face_inner_id = cube_id % face_count;

        int local_coords[3];
        local_coords[faceAxisTable[face_id][1]] = faceAxisTable[face_id][0] * stride;
        local_coords[faceAxisTable[face_id][2]] = face_inner_id % (stride - 1) + 1;
        local_coords[faceAxisTable[face_id][3]] = int(face_inner_id / (stride - 1)) + 1;

        uint64_t pos_hash = hash3(primal_base[primal_id][0] + local_coords[0],
                         primal_base[primal_id][1] + local_coords[1],
                         primal_base[primal_id][2] + local_coords[2]);
        uint64_t dc_pos = hashtable_lookup(
                param.d_key, param.d_val, param.size, param.config, param.num_funcs, param.num_buckets,
                pos_hash);
        if (dc_pos == EMPTY_CELL) {
            return;
        }
        int cnt = atomicAdd(acc_inds_cnt + dc_pos - 1, 4);
        if (cnt >= 8) {
            printf("Topology Error at depth %d (Face), primal_id = %d\n", d, primal_id);
            return;
        }
        for (int ai = 0; ai < 4; ai++) {
            acc_inds[dc_pos - 1][faceAccIndsTable[face_id][ai]][0] = primal_id + idx_base;
            acc_inds[dc_pos - 1][faceAccIndsTable[face_id][ai]][1] = d;
        }
    } else if (cube_id >= cube_corner_count) {
        // Edge attached, should accumulate 2 MC inds if DC exists.
        cube_id -= cube_corner_count;
        int edge_id = cube_id / (stride - 1);
        int edge_inner_id = cube_id % (stride - 1);

        int local_coords[3];
        local_coords[edgeAxisTable[edge_id][2]] = edgeAxisTable[edge_id][0] * stride;
        local_coords[edgeAxisTable[edge_id][3]] = edgeAxisTable[edge_id][1] * stride;
        local_coords[edgeAxisTable[edge_id][4]] = edge_inner_id + 1;

        uint64_t pos_hash = hash3(primal_base[primal_id][0] + local_coords[0],
                                  primal_base[primal_id][1] + local_coords[1],
                                  primal_base[primal_id][2] + local_coords[2]);
        uint64_t dc_pos = hashtable_lookup(
                param.d_key, param.d_val, param.size, param.config, param.num_funcs, param.num_buckets,
                pos_hash);
        if (dc_pos == EMPTY_CELL) {
            return;
        }
        int cnt = atomicAdd(acc_inds_cnt + dc_pos - 1, 2);
        if (cnt >= 8) {
            printf("Topology Error at depth %d (Edge), primal_id = %d\n", d, primal_id);
            return;
        }
        for (int ai = 0; ai < 2; ai++) {
            acc_inds[dc_pos - 1][edgeAccIndsTable[edge_id][ai]][0] = primal_id + idx_base;
            acc_inds[dc_pos - 1][edgeAccIndsTable[edge_id][ai]][1] = d;
        }
    } else {
        // Corner attached, should just accumulate 1 (not expanding).
        uint64_t pos_hash = hash3(primal_base[primal_id][0] + cornerAxisTable[cube_id][0] * stride,
                                  primal_base[primal_id][1] + cornerAxisTable[cube_id][1] * stride,
                                  primal_base[primal_id][2] + cornerAxisTable[cube_id][2] * stride);
        uint64_t dc_pos = hashtable_lookup(
                param.d_key, param.d_val, param.size, param.config, param.num_funcs, param.num_buckets,
                pos_hash);
        if (dc_pos == EMPTY_CELL) {
            return;
        }
        int cnt = atomicAdd(acc_inds_cnt + dc_pos - 1, 1);
        if (cnt >= 8) {
            printf("Topology Error at depth %d (Corner), primal_id = %d\n", d, primal_id);
            return;
        }
        acc_inds[dc_pos - 1][cornerAccIndsTable[cube_id]][0] = primal_id + idx_base;
        acc_inds[dc_pos - 1][cornerAccIndsTable[cube_id]][1] = d;
    }
}

std::vector<torch::Tensor> accumulate_dual_corner_indices(
        const HashLookupData &dual_centers_hash_data, py::dict expand_primal_base, py::dict primal_strides,
        py::dict primal_inds_base) {

    auto hash_param = dual_centers_hash_data.get_param();
    int n_dual_centers = dual_centers_hash_data.inserted_size;

//    auto long_tensor_option = at::device(dual_centers_hash.device()).dtype(at::ScalarType::Long);
    auto int_tensor_option = at::device(dual_centers_hash_data.device()).dtype(at::ScalarType::Int);

    torch::Tensor acc_inds = torch::zeros({n_dual_centers, 8, 2}, int_tensor_option);
    torch::Tensor acc_inds_cnt = torch::zeros(n_dual_centers, int_tensor_option);

    // Iterate over d, and 6 x (2 ** scale) x (2 ** scale) x (2 ** scale) values.
    /**
     * Future notes for faster computation:
     *  Currently, for voxels with scale, we test all possibilities, i.e. all subsamples on the faces.
     *  If we have a face-adjacent tree, we could always limit num_values to stride==2
     */
    for (auto d_stride = primal_strides.begin(); d_stride != primal_strides.end(); ++d_stride) {
        int stride = d_stride->second.cast<int>();
        int index_base = primal_inds_base[d_stride->first].cast<int>();
//        printf("OK: %d, %d\n", stride, d_stride->first.cast<int>());
        torch::Tensor primal_base = expand_primal_base[d_stride->first].cast<torch::Tensor>();
        int num_primal = primal_base.size(0);
        int num_values = ((stride + 1) * (stride + 1) * 2 + (stride - 1) * stride * 4) * num_primal;
        if (num_values > 0) {
            dim3 dimBlock = dim3(256);
            dim3 dimGrid = dim3((num_values + dimBlock.x - 1) / dimBlock.x);
            acc_inds_kernel<<<dimGrid, dimBlock>>>(
                    num_primal, stride, d_stride->first.cast<int>(), index_base,
                    hash_param, primal_base.int2in(), acc_inds.int3in(), acc_inds_cnt.data<int>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }

    dual_centers_hash_data.release_param(hash_param);

    return {acc_inds, acc_inds_cnt};
}
