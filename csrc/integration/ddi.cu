#include <torch/torch.h>
#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include "../sparse_op/atomics.cuh"

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32 tensor");
#define CHECK_LONG(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Long, #x " must be long tensor");
#define CHECK_1DIM(x) TORCH_CHECK(x.ndimension() == 1, #x " must be 1-dim tensor");

#define CHECK_LONG_INPUT(x) CHECK_CUDA(x); CHECK_LONG(x);
#define CHECK_FLOAT_INPUT(x) CHECK_CUDA(x); CHECK_FLOAT(x);

using LongAccessor = torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits>;
using Long2Accessor = torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits>;
using FloatAccessor = torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits>;
using Float2Accessor = torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>;
using Float3Accessor = torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits>;

__global__ static void ddi_forward_kernel(
        long n_pairs,
        const Float2Accessor source_expanded, const Float2Accessor target_expanded,
        const Long2Accessor rel_pos,
        const Float3Accessor f_f_i, const Float3Accessor df_df_i,
        float* p_mats, float* q_mats) {

    int pair_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_id >= n_pairs) {
        return;
    }

    int n_components = f_f_i.size(0);
    int si = threadIdx.y;
    if (si >= n_components) {
        return;
    }

    for (int ti = 0; ti < n_components; ++ti) {
        #pragma unroll
        for (int v = 0; v < 3; ++ v) {
            float st = source_expanded[pair_id][si * 3 + v] * target_expanded[pair_id][ti * 3 + v];
            int r_pos = (int) rel_pos[pair_id][v];
            // No need for atomic operations
            atomAdd(p_mats + pair_id * 3 + v, st * f_f_i[si][ti][r_pos]);
            atomAdd(q_mats + pair_id * 3 + v, st * df_df_i[si][ti][r_pos]);
        }
    }
}

std::pair<torch::Tensor, torch::Tensor> ddi_forward(
        const torch::Tensor& source_expanded, const torch::Tensor& target_expanded,
        const torch::Tensor& rel_pos,
        const torch::Tensor& f_f_i, const torch::Tensor& df_df_i) {

    long n_pairs = source_expanded.size(0);
    torch::Tensor p_mats = torch::zeros({n_pairs, 3},
                                        torch::dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor q_mats = torch::zeros({n_pairs, 3},
                                        torch::dtype(torch::kFloat32).device(torch::kCUDA));

    dim3 dimBlock = dim3(32, 16);
    dim3 dimGrid = dim3((n_pairs + dimBlock.x - 1) / dimBlock.x);
    ddi_forward_kernel<<<dimGrid, dimBlock>>>(n_pairs,
                                              source_expanded.packed_accessor32<float , 2, torch::RestrictPtrTraits>(),
                                              target_expanded.packed_accessor32<float , 2, torch::RestrictPtrTraits>(),
                                              rel_pos.packed_accessor32<int64_t , 2, torch::RestrictPtrTraits>(),
                                              f_f_i.packed_accessor32<float , 3, torch::RestrictPtrTraits>(),
                                              df_df_i.packed_accessor32<float , 3, torch::RestrictPtrTraits>(),
                                              p_mats.data_ptr<float>(),
                                              q_mats.data_ptr<float>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_pair(p_mats, q_mats);
}

__global__ static void ddi_backward_kernel(
        long n_pairs,
        const Float2Accessor source_expanded, const Float2Accessor target_expanded,
        const Long2Accessor rel_pos,
        const Float3Accessor f_f_i, const Float3Accessor df_df_i,
        const Float2Accessor grad_p, const Float2Accessor grad_q,
        float* grad_s, float* grad_t) {

    int pair_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_id >= n_pairs) {
        return;
    }

    int n_components = f_f_i.size(0);
    int si = threadIdx.y;
    if (si >= n_components) {
        return;
    }

    for (int ti = 0; ti < n_components; ++ti) {
        #pragma unroll
        for (int v = 0; v < 3; ++ v) {
            int r_pos = (int) rel_pos[pair_id][v];
            float grad_const_pq = grad_p[pair_id][v] * f_f_i[si][ti][r_pos] +
                    grad_q[pair_id][v] * df_df_i[si][ti][r_pos];
            // No need for atomic operations
            atomAdd(grad_s + pair_id * n_components * 3 + si * 3 + v,
                    grad_const_pq * target_expanded[pair_id][ti * 3 + v]);
            atomAdd(grad_t + pair_id * n_components * 3 + ti * 3 + v,
                    grad_const_pq * source_expanded[pair_id][si * 3 + v]);
        }
    }

}

std::pair<torch::Tensor, torch::Tensor> ddi_backward(
        const torch::Tensor& source_expanded, const torch::Tensor& target_expanded,
        const torch::Tensor& rel_pos,
        const torch::Tensor& f_f_i, const torch::Tensor& df_df_i,
        const torch::Tensor& grad_p, const torch::Tensor& grad_q) {

    long n_pairs = source_expanded.size(0);
    torch::Tensor grad_s = torch::zeros({n_pairs, 3 * f_f_i.size(0)},
                                        torch::dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor grad_t = torch::zeros({n_pairs, 3 * f_f_i.size(0)},
                                        torch::dtype(torch::kFloat32).device(torch::kCUDA));

    dim3 dimBlock = dim3(32, 16);
    dim3 dimGrid = dim3((n_pairs + dimBlock.x - 1) / dimBlock.x);
    ddi_backward_kernel<<<dimGrid, dimBlock>>>(n_pairs,
                                              source_expanded.packed_accessor32<float , 2, torch::RestrictPtrTraits>(),
                                              target_expanded.packed_accessor32<float , 2, torch::RestrictPtrTraits>(),
                                              rel_pos.packed_accessor32<int64_t , 2, torch::RestrictPtrTraits>(),
                                              f_f_i.packed_accessor32<float , 3, torch::RestrictPtrTraits>(),
                                              df_df_i.packed_accessor32<float , 3, torch::RestrictPtrTraits>(),
                                               grad_p.packed_accessor32<float , 2, torch::RestrictPtrTraits>(),
                                               grad_q.packed_accessor32<float , 2, torch::RestrictPtrTraits>(),
                                               grad_s.data_ptr<float>(),
                                               grad_t.data_ptr<float>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_pair(grad_s, grad_t);
}
