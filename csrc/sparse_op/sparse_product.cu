#include <torch/torch.h>
#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include "hashmap_cuda.cuh"
#include "atomics.cuh"

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32 tensor");
#define CHECK_LONG(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Long, #x " must be long tensor");
#define CHECK_1DIM(x) TORCH_CHECK(x.ndimension() == 1, #x " must be 1-dim tensor");

#define CHECK_LONG_INPUT(x) CHECK_CUDA(x); CHECK_LONG(x); CHECK_1DIM(x);
#define CHECK_FLOAT_INPUT(x) CHECK_CUDA(x); CHECK_FLOAT(x);

using LongAccessor = torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits>;
using FloatAccessor = torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits>;

__global__ static void fill_meshgrid(size_t n_batch,
        const LongAccessor a_cumsum, const LongAccessor b_cumsum, const LongAccessor ab_cumsum,
        LongAccessor a_inds, LongAccessor b_inds) {
    unsigned int batch_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_id >= n_batch) {
        return;
    }

    int64_t a_start = batch_id == 0 ? 0 : a_cumsum[batch_id - 1];
    int64_t a_end = a_cumsum[batch_id];
    int64_t b_start = batch_id == 0 ? 0 : b_cumsum[batch_id - 1];
    int64_t b_end = b_cumsum[batch_id];
    int64_t ind_base = batch_id == 0 ? 0 : ab_cumsum[batch_id - 1];

    int idx = 0;

//    for (int64_t a = a_start; a < a_end; ++a) {
//        for (int64_t b = b_start; b < b_end; ++b, ++idx) {
//            a_inds[ind_base + idx] = a;
//            b_inds[ind_base + idx] = b;
//        }
//    }

    unsigned int a = threadIdx.y + a_start;
    if (a >= a_end) return;
    ind_base += threadIdx.y * (b_end - b_start);
    for (int64_t b = b_start; b < b_end; ++b, ++idx) {
        a_inds[ind_base + idx] = a;
        b_inds[ind_base + idx] = b;
    }
}

std::pair<torch::Tensor, torch::Tensor> sparse_meshgrid(const torch::Tensor& a_lengths, const torch::Tensor& b_lengths) {
    // Check long ternsor
    CHECK_LONG_INPUT(a_lengths);
    CHECK_LONG_INPUT(b_lengths);

    torch::Tensor ab = a_lengths * b_lengths;
    torch::Tensor a_cumsum = torch::cumsum(a_lengths, 0);
    torch::Tensor b_cumsum = torch::cumsum(b_lengths, 0);
    torch::Tensor ab_cumsum = torch::cumsum(ab, 0);

    auto n_length = ab_cumsum[-1].item<int64_t>();
    torch::Tensor a_inds = torch::empty(n_length,
                                        torch::dtype(torch::kLong).device(torch::kCUDA));
    torch::Tensor b_inds = torch::empty(n_length,
                                        torch::dtype(torch::kLong).device(torch::kCUDA));

    size_t n_batch = a_lengths.size(0);

    dim3 dimBlock = dim3(16, 32);
    dim3 dimGrid = dim3((n_batch + dimBlock.x - 1) / dimBlock.x);
    fill_meshgrid<<<dimGrid, dimBlock>>>(n_batch,
                                         a_cumsum.packed_accessor32<int64_t , 1, torch::RestrictPtrTraits>(),
                                         b_cumsum.packed_accessor32<int64_t , 1, torch::RestrictPtrTraits>(),
                                         ab_cumsum.packed_accessor32<int64_t , 1, torch::RestrictPtrTraits>(),
                                         a_inds.packed_accessor32<int64_t , 1, torch::RestrictPtrTraits>(),
                                         b_inds.packed_accessor32<int64_t , 1, torch::RestrictPtrTraits>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_pair(a_inds, b_inds);
}

__device__ static uint64_t hash2(unsigned int a, unsigned int b) {
    uint64_t hash = 14695981039346656037UL;
    hash ^= a;
    hash *= 1099511628211UL;
    hash ^= b;
    hash *= 1099511628211UL;
    hash = (hash >> 60) ^ (hash & 0xFFFFFFFFFFFFFFF);
    return hash;
}

__global__ static void hash_kernel(size_t n_elem, const LongAccessor a, const LongAccessor b, LongAccessor res) {
    unsigned int elem_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (elem_id >= n_elem) {
        return;
    }
    res[elem_id] = hash2((unsigned int) a[elem_id], (unsigned int) b[elem_id]);
}

// Dispatch multiplication
__inline__ __device__ static float val_mult(
        const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> a_val,
        const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> b_val,
        int64_t a, int64_t b) {
    return a_val[a] * b_val[b];
}

__inline__ __device__ static float val_mult(
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> a_val,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> b_val,
        int64_t a, int64_t b) {
    return a_val[a][0] * b_val[b][0] + a_val[a][1] * b_val[b][1] + a_val[a][2] * b_val[b][2];
}

template <int Dim>
__global__ static void smult_kernel(size_t n_points, HashLookupParam param,
                                    const torch::PackedTensorAccessor32<float, Dim, torch::RestrictPtrTraits> a_val,
                                    const torch::PackedTensorAccessor32<float, Dim, torch::RestrictPtrTraits> b_val,
                                    const LongAccessor a_inds, const LongAccessor b_inds,
                                    const LongAccessor a_cumsum, const LongAccessor b_cumsum,
                                    const FloatAccessor weight,
                                    float* res) {
    unsigned int batch_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_id >= n_points) {
        return;
    }

    int64_t a_start = batch_id == 0 ? 0 : a_cumsum[batch_id - 1];
    int64_t a_end = a_cumsum[batch_id];
    int64_t b_start = batch_id == 0 ? 0 : b_cumsum[batch_id - 1];
    int64_t b_end = b_cumsum[batch_id];

    unsigned int a = threadIdx.y + a_start;
    if (a >= a_end) return;

    float w = weight[batch_id];

    for (int64_t b = b_start; b < b_end; ++b) {
        // Compute Multiplication, Look into hash table and atomic add to correct position.
        float fg = val_mult(a_val, b_val, a, b) * w;
        uint64_t pos_hash = hash2((unsigned int) a_inds[a], (unsigned int) b_inds[b]);
        uint64_t add_pos = hashtable_lookup(param, pos_hash);
        if (add_pos == EMPTY_CELL) {
//            printf("[smult_kernel assert error] Lookup value of %d, %d failed in HashTable.\n",
//                   (unsigned int) a_inds[a], (unsigned int) b_inds[b]);
            continue;
        }
        atomAdd(res + add_pos - 1, fg);
    }
}

__inline__ __device__ static void val_mult_back(
        const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> a_val,
        const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> b_val,
        int64_t a, int64_t b, float* grad_a, float* grad_b, float grad_fg) {
    atomAdd(grad_a + a, grad_fg * b_val[b]);
    atomAdd(grad_b + b, grad_fg * a_val[a]);
}

__inline__ __device__ static void val_mult_back(
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> a_val,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> b_val,
        int64_t a, int64_t b, float* grad_a, float* grad_b, float grad_fg) {
    atomAdd(grad_a + a * 3 + 0, grad_fg * b_val[b][0]);
    atomAdd(grad_a + a * 3 + 1, grad_fg * b_val[b][1]);
    atomAdd(grad_a + a * 3 + 2, grad_fg * b_val[b][2]);
    atomAdd(grad_b + b * 3 + 0, grad_fg * a_val[a][0]);
    atomAdd(grad_b + b * 3 + 1, grad_fg * a_val[a][1]);
    atomAdd(grad_b + b * 3 + 2, grad_fg * a_val[a][2]);
}

template <int Dim>
__global__ static void smult_backward_kernel(size_t n_points, HashLookupParam param,
                                             const torch::PackedTensorAccessor32<float, Dim, torch::RestrictPtrTraits> a_val,
                                             const torch::PackedTensorAccessor32<float, Dim, torch::RestrictPtrTraits> b_val,
                                    const LongAccessor a_inds, const LongAccessor b_inds,
                                    const LongAccessor a_cumsum, const LongAccessor b_cumsum, const FloatAccessor weight,
                                    const FloatAccessor grad_res, float* grad_a, float* grad_b, float* grad_w) {
    unsigned int batch_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_id >= n_points) {
        return;
    }

    int64_t a_start = batch_id == 0 ? 0 : a_cumsum[batch_id - 1];
    int64_t a_end = a_cumsum[batch_id];
    int64_t b_start = batch_id == 0 ? 0 : b_cumsum[batch_id - 1];
    int64_t b_end = b_cumsum[batch_id];

    unsigned int a = threadIdx.y + a_start;
    if (a >= a_end) return;

    float w = weight[batch_id];

    for (int64_t b = b_start; b < b_end; ++b) {
        uint64_t pos_hash = hash2((unsigned int) a_inds[a], (unsigned int) b_inds[b]);
        uint64_t add_pos = hashtable_lookup(param, pos_hash);
        if (add_pos == EMPTY_CELL) {
//            printf("[smult_backward_kernel assert error] Lookup value of %d, %d failed in HashTable.\n",
//                   (unsigned int) a_inds[a], (unsigned int) b_inds[b]);
            continue;
        }
        float grad_fg = grad_res[add_pos - 1];
        float fg = val_mult(a_val, b_val, a, b);
        val_mult_back(a_val, b_val, a, b, grad_a, grad_b, grad_fg * w);
        atomAdd(grad_w + batch_id, grad_fg * fg);
    }
}

// This is a fully fused implementation that takes in all data, and outputs desired entries in the sparse matrix.
//      Hopefully we can make execution and memory better.
std::pair<torch::Tensor, HashLookupData> screened_multiplication(
        const torch::Tensor& src_ids, const torch::Tensor& tgt_ids,
        const torch::Tensor& a_val, const torch::Tensor& b_val,
        const torch::Tensor& a_inds, const torch::Tensor& b_inds,
        const torch::Tensor& a_lengths, const torch::Tensor& b_lengths,
        const torch::Tensor& weight) {
    CHECK_LONG_INPUT(src_ids); CHECK_LONG_INPUT(tgt_ids);
    CHECK_FLOAT_INPUT(a_val); CHECK_FLOAT_INPUT(b_val);
    CHECK_LONG_INPUT(a_inds); CHECK_LONG_INPUT(b_inds);
    CHECK_LONG_INPUT(a_lengths); CHECK_LONG_INPUT(b_lengths);
    CHECK_FLOAT_INPUT(weight);

    auto long_tensor_option = at::device(src_ids.device()).dtype(at::ScalarType::Long);
    auto float_tensor_option = at::device(src_ids.device()).dtype(at::ScalarType::Float);

    size_t n_source = src_ids.size(0);
    torch::Tensor source_hash = torch::empty(n_source, long_tensor_option);
    // Build Hash Table.
    {
        dim3 dimBlock = dim3(512);
        dim3 dimGrid = dim3((n_source + dimBlock.x - 1) / dimBlock.x);
        hash_kernel<<<dimGrid, dimBlock>>>(n_source,
                                            src_ids.packed_accessor32<int64_t , 1, torch::RestrictPtrTraits>(),
                                            tgt_ids.packed_accessor32<int64_t , 1, torch::RestrictPtrTraits>(),
                                           source_hash.packed_accessor32<int64_t , 1, torch::RestrictPtrTraits>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    const int nextPow2 = pow(2, ceil(log2((double)n_source)));
//    int table_size = (n_source < 2048) ? 4 * nextPow2 : 2 * nextPow2;
    int table_size = 4 * nextPow2;
    if (table_size < 512) { table_size = 512; }
    int num_funcs = 3;
    CuckooHashTableCuda_Multi in_hash_table(table_size, 8 * ceil(log2((double)n_source)),
                                            num_funcs);
    at::Tensor key_buf = torch::zeros({table_size}, long_tensor_option);
    at::Tensor val_buf = torch::zeros({table_size},long_tensor_option);
    at::Tensor hash_key = torch::zeros({num_funcs * table_size}, long_tensor_option);
    at::Tensor hash_val = torch::zeros({num_funcs * table_size}, long_tensor_option);

    in_hash_table.insert_vals((uint64_t *)(source_hash.data_ptr<int64_t>()), nullptr,
                              (uint64_t *)(key_buf.data_ptr<int64_t>()),
                              (uint64_t *)(val_buf.data_ptr<int64_t>()),
                              (uint64_t *)(hash_key.data_ptr<int64_t>()),
                              (uint64_t *)(hash_val.data_ptr<int64_t>()), n_source);

    auto hash_data = in_hash_table.get_data(hash_key, hash_val);
    auto hash_param = hash_data.get_param();

    // Distribute values.
    torch::Tensor a_cumsum = torch::cumsum(a_lengths, 0);
    torch::Tensor b_cumsum = torch::cumsum(b_lengths, 0);
    size_t n_points = a_lengths.size(0);

    torch::Tensor res = torch::zeros(n_source, float_tensor_option);
    {
        dim3 dimBlock = dim3(16, 32);
        dim3 dimGrid = dim3((n_points + dimBlock.x - 1) / dimBlock.x);
        if (a_val.ndimension() == 1) {
            smult_kernel<1><<<dimGrid, dimBlock>>>(n_points, hash_param,
                                                a_val.packed_accessor32<float , 1, torch::RestrictPtrTraits>(),
                                                b_val.packed_accessor32<float , 1, torch::RestrictPtrTraits>(),
                                                a_inds.packed_accessor32<int64_t , 1, torch::RestrictPtrTraits>(),
                                                b_inds.packed_accessor32<int64_t , 1, torch::RestrictPtrTraits>(),
                                                a_cumsum.packed_accessor32<int64_t , 1, torch::RestrictPtrTraits>(),
                                                b_cumsum.packed_accessor32<int64_t , 1, torch::RestrictPtrTraits>(),
                                                weight.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                                                res.data_ptr<float>()
            );
        } else {
            smult_kernel<2><<<dimGrid, dimBlock>>>(n_points, hash_param,
                                                a_val.packed_accessor32<float , 2, torch::RestrictPtrTraits>(),
                                                b_val.packed_accessor32<float , 2, torch::RestrictPtrTraits>(),
                                                a_inds.packed_accessor32<int64_t , 1, torch::RestrictPtrTraits>(),
                                                b_inds.packed_accessor32<int64_t , 1, torch::RestrictPtrTraits>(),
                                                a_cumsum.packed_accessor32<int64_t , 1, torch::RestrictPtrTraits>(),
                                                b_cumsum.packed_accessor32<int64_t , 1, torch::RestrictPtrTraits>(),
                                                weight.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                                                res.data_ptr<float>()
            );
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    hash_data.release_param(hash_param);

    return std::make_pair(res, hash_data);
}

std::vector<torch::Tensor> screened_multiplication_backward(const HashLookupData& hash_data,
                                                   const torch::Tensor& a_val, const torch::Tensor& b_val,
                                                   const torch::Tensor& a_inds, const torch::Tensor& b_inds,
                                                   const torch::Tensor& a_lengths, const torch::Tensor& b_lengths,
                                                   const torch::Tensor& weight,
                                                   const torch::Tensor& grad_res) {
    CHECK_FLOAT_INPUT(a_val); CHECK_FLOAT_INPUT(b_val);
    CHECK_LONG_INPUT(a_inds); CHECK_LONG_INPUT(b_inds);
    CHECK_LONG_INPUT(a_lengths); CHECK_LONG_INPUT(b_lengths);

    auto float_tensor_option = at::device(grad_res.device()).dtype(at::ScalarType::Float);

    size_t n_source = grad_res.size(0);
    auto hash_param = hash_data.get_param();

    // Distribute values.
    torch::Tensor a_cumsum = torch::cumsum(a_lengths, 0);
    torch::Tensor b_cumsum = torch::cumsum(b_lengths, 0);
    size_t n_points = a_lengths.size(0);

    torch::Tensor grad_a = torch::zeros(a_val.sizes(), float_tensor_option);
    torch::Tensor grad_b = torch::zeros(b_val.sizes(), float_tensor_option);
    torch::Tensor grad_w = torch::zeros(weight.size(0), float_tensor_option);
    {
        dim3 dimBlock = dim3(16, 32);
        dim3 dimGrid = dim3((n_points + dimBlock.x - 1) / dimBlock.x);
        if (a_val.ndimension() == 1) {
            smult_backward_kernel<1><<<dimGrid, dimBlock>>>(n_points, hash_param,
                                                            a_val.packed_accessor32<float , 1, torch::RestrictPtrTraits>(),
                                                            b_val.packed_accessor32<float , 1, torch::RestrictPtrTraits>(),
                                                            a_inds.packed_accessor32<int64_t , 1, torch::RestrictPtrTraits>(),
                                                            b_inds.packed_accessor32<int64_t , 1, torch::RestrictPtrTraits>(),
                                                            a_cumsum.packed_accessor32<int64_t , 1, torch::RestrictPtrTraits>(),
                                                            b_cumsum.packed_accessor32<int64_t , 1, torch::RestrictPtrTraits>(),
                                                            weight.packed_accessor32<float , 1, torch::RestrictPtrTraits>(),
                                                            grad_res.packed_accessor32<float , 1, torch::RestrictPtrTraits>(),
                                                            grad_a.data_ptr<float>(), grad_b.data_ptr<float>(),
                                                                    grad_w.data_ptr<float>()
            );
        } else {
            smult_backward_kernel<2><<<dimGrid, dimBlock>>>(n_points, hash_param,
                                                            a_val.packed_accessor32<float , 2, torch::RestrictPtrTraits>(),
                                                            b_val.packed_accessor32<float , 2, torch::RestrictPtrTraits>(),
                                                            a_inds.packed_accessor32<int64_t , 1, torch::RestrictPtrTraits>(),
                                                            b_inds.packed_accessor32<int64_t , 1, torch::RestrictPtrTraits>(),
                                                            a_cumsum.packed_accessor32<int64_t , 1, torch::RestrictPtrTraits>(),
                                                            b_cumsum.packed_accessor32<int64_t , 1, torch::RestrictPtrTraits>(),
                                                            weight.packed_accessor32<float , 1, torch::RestrictPtrTraits>(),
                                                            grad_res.packed_accessor32<float , 1, torch::RestrictPtrTraits>(),
                                                            grad_a.data_ptr<float>(), grad_b.data_ptr<float>(),
                                                                    grad_w.data_ptr<float>()
            );
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    hash_data.release_param(hash_param);

    return {grad_a, grad_b, grad_w};
}
