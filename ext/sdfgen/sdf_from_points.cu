#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include "../../csrc/common/kdtree_cuda.cuh"
#include "../../csrc/common/cutil_math.h"

using DataAccessor = torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>;
using SDFAccessor = torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits>;

class ThrustAllocator {
public:
    typedef char value_type;

    char* allocate(std::ptrdiff_t size) {
        return static_cast<char*>(c10::cuda::CUDACachingAllocator::raw_alloc(size));
    }

    void deallocate(char* p, size_t size) {
        c10::cuda::CUDACachingAllocator::raw_delete(p);
    }
};

__global__ static void ComputeSDFKernel(size_t num_samples, int num_votes,
                                        const DataAccessor ref_xyz, const DataAccessor ref_normals,
                                        const int* __restrict__ knn_index,
                                        const DataAccessor query_xyz,
                                        float stdv, SDFAccessor sdf_val) {
    unsigned int sample_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_id >= num_samples) {
        return;
    }

    float3 sample_pos = make_float3(query_xyz[sample_id][0], query_xyz[sample_id][1], query_xyz[sample_id][2]);

    float sdf;
    int num_pos = 0;
    for (int vote_i = 0; vote_i < num_votes; ++vote_i) {
        int cur_ind = knn_index[sample_id * num_votes + vote_i];
        float3 nb_pos = make_float3(ref_xyz[cur_ind][0], ref_xyz[cur_ind][1], ref_xyz[cur_ind][2]);
        float3 nb_normal = make_float3(ref_normals[cur_ind][0], ref_normals[cur_ind][1], ref_normals[cur_ind][2]);
        float3 ray_vec = sample_pos - nb_pos;

        float d = dot(nb_normal, ray_vec);
        if (vote_i == 0) {
            float ray_vec_len = length(ray_vec);
            if (ray_vec_len < stdv) {
                sdf = abs(d);
            } else {
                sdf = ray_vec_len;
            }
        }
        if (d > 0) { num_pos += 1; }
    }

    if (num_pos <= num_votes / 2) {
        sdf_val[sample_id] = -sdf;
    } else {
        sdf_val[sample_id] = sdf;
    }
}

torch::Tensor sdf_from_points(const torch::Tensor& queries, const torch::Tensor& ref_xyz, const torch::Tensor& ref_normal,
                              int nb_points, float stdv) {
    CHECK_CUDA(queries); CHECK_IS_FLOAT(queries)
    CHECK_CUDA(ref_xyz);
    CHECK_CUDA(ref_normal);

    // Index requires reference to have stride 4.
    torch::Tensor strided_ref = ref_xyz;
    if (ref_xyz.stride(0) != 4) {
        strided_ref = torch::zeros({strided_ref.size(0), 4}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        strided_ref.index_put_({torch::indexing::Ellipsis, torch::indexing::Slice(torch::indexing::None, 3)}, ref_xyz);
    }

    // Note: if needed, should refer to this:
//    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
//    ThrustAllocator allocator;
//    auto policy = thrust::cuda::par(allocator).on(stream);

    // Build KDTree based on reference point cloud
    size_t n_ref = strided_ref.size(0);
    tinyflann::KDTreeCuda3dIndex<tinyflann::CudaL2> knn_index(strided_ref.data_ptr<float>(), n_ref);
    knn_index.buildIndex();

    // Compute for each point its nearest N neighbours.
    size_t n_query = queries.size(0);
    thrust::device_vector<float> dist(n_query * nb_points);
    thrust::device_vector<int> indices(n_query * nb_points);
    knn_index.knnSearch(queries.data_ptr<float>(), n_query, queries.stride(0), (int*) indices.data().get(),
                        (float*) dist.data().get(), nb_points);

    // Compute sdf value.
    torch::Tensor sdf = torch::zeros(n_query, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    {
        dim3 dimBlock = dim3(256);
        dim3 dimGrid = dim3((n_query + dimBlock.x - 1) / dimBlock.x);
        ComputeSDFKernel<<<dimGrid, dimBlock>>>(n_query, nb_points,
                                                ref_xyz.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                                                ref_normal.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                                                indices.data().get(),
                                                queries.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                                                stdv,
                                                sdf.packed_accessor32<float, 1, torch::RestrictPtrTraits>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return sdf;
}
