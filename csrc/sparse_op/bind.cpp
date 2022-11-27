#include <torch/extension.h>
#include "../common/hashmap_cuda.cuh"

std::pair<torch::Tensor, torch::Tensor> sparse_meshgrid(const torch::Tensor& a_lengths, const torch::Tensor& b_lengths);
torch::Tensor screened_multiplication(const HashLookupData& hash_data,
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

void convolution_forward_cuda(at::Tensor in_feat, at::Tensor out_feat,
                              at::Tensor kernel, at::Tensor neighbor_map,
                              at::Tensor neighbor_offset, const bool transpose, const bool no_acc);

void convolution_backward_cuda(at::Tensor in_feat, at::Tensor grad_in_feat,
                               at::Tensor grad_out_feat, at::Tensor kernel,
                               at::Tensor grad_kernel, at::Tensor neighbor_map,
                               at::Tensor neighbor_offset,
                               const bool transpose);

torch::Tensor plane_fitting(torch::Tensor xx, torch::Tensor xy, torch::Tensor xz,
                            torch::Tensor yy, torch::Tensor yz, torch::Tensor zz);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("plane_fitting", &plane_fitting, "Fit plane.");
    m.def("sparse_meshgrid", &sparse_meshgrid, "Compute a sparse meshgrid for indexing.");
    m.def("screened_multiplication", &screened_multiplication, "Heavily fused operation.");
    m.def("screened_multiplication_backward", &screened_multiplication_backward, "Heavily fused operation.");
    m.def("convolution_forward_cuda", &convolution_forward_cuda);
    m.def("convolution_backward_cuda", &convolution_backward_cuda);
}
