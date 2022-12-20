#include <torch/extension.h>

// GPU solver: cusparse
torch::Tensor solve_cusparse(const torch::Tensor& Ap, const torch::Tensor& Aj, const torch::Tensor& Ax, const torch::Tensor& b, float tol);
void init_cusolver_handle();

std::pair<torch::Tensor, int> solve_pcg_diag(
        const torch::Tensor& Ap, const torch::Tensor& Aj, const torch::Tensor& Ax, const torch::Tensor& b,
        const torch::Tensor& inv_diag_A,
        const float tol, const int max_iter, const bool res_fix);

// Let's dispatch device outside, due to possibly different interfaces.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("solve_cusparse", &solve_cusparse, "Solve sparse matrix using CU-Sparse LU.");
    m.def("solve_pcg_diag", &solve_pcg_diag, "Solve sparse matrix using PCG.");
    m.def("init_cusolver", &init_cusolver_handle, "Must be called before calling cusolver.");
}
