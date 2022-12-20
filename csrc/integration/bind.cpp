#include <torch/extension.h>

std::pair<torch::Tensor, torch::Tensor> ddi_forward(
        const torch::Tensor& source_expanded, const torch::Tensor& target_expanded,
        const torch::Tensor& rel_pos,
        const torch::Tensor& f_f_i, const torch::Tensor& df_df_i);

std::pair<torch::Tensor, torch::Tensor> ddi_backward(
        const torch::Tensor& source_expanded, const torch::Tensor& target_expanded,
        const torch::Tensor& rel_pos,
        const torch::Tensor& f_f_i, const torch::Tensor& df_df_i,
        const torch::Tensor& grad_p, const torch::Tensor& grad_q);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ddi_forward", &ddi_forward, "DDI Forward");
    m.def("ddi_backward", &ddi_backward, "DDI Backward");
}
