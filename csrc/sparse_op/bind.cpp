#include <torch/extension.h>
#include "hashmap_cuda.cuh"

std::pair<torch::Tensor, torch::Tensor> sparse_meshgrid(const torch::Tensor& a_lengths, const torch::Tensor& b_lengths);
std::pair<torch::Tensor, HashLookupData> screened_multiplication(const torch::Tensor& src_ids, const torch::Tensor& tgt_ids,
                                                                 const torch::Tensor& a_val, const torch::Tensor& b_val,
                                                                 const torch::Tensor& a_inds, const torch::Tensor& b_inds,
                                                                 const torch::Tensor& a_lengths, const torch::Tensor& b_lengths,
                                                                 const torch::Tensor& weight);
std::vector<torch::Tensor> screened_multiplication_backward(const HashLookupData& hash_data,
                                                            const torch::Tensor& a_val, const torch::Tensor& b_val,
                                                            const torch::Tensor& a_inds, const torch::Tensor& b_inds,
                                                            const torch::Tensor& a_lengths, const torch::Tensor& b_lengths,
                                                            const torch::Tensor& weight,
                                                            const torch::Tensor& grad_res);

torch::Tensor plane_fitting(torch::Tensor xx, torch::Tensor xy, torch::Tensor xz,
                            torch::Tensor yy, torch::Tensor yz, torch::Tensor zz);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("plane_fitting", &plane_fitting, "Fit plane.");
    m.def("sparse_meshgrid", &sparse_meshgrid, "Compute a sparse meshgrid for indexing.");
    m.def("screened_multiplication", &screened_multiplication, "Heavily fused operation.");
    m.def("screened_multiplication_backward", &screened_multiplication_backward, "Heavily fused operation.");
    py::class_<HashLookupData>(m, "HashLookupData");
}
