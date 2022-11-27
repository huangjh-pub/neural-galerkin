#include <torch/extension.h>
//#include <Eigen/Dense>

torch::Tensor sdf_from_points(const torch::Tensor& queries,
                              const torch::Tensor& ref_xyz,
                              const torch::Tensor& ref_normal,
                              int nb_points, float stdv);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sdf_from_points", &sdf_from_points, "Compute sdf value from reference points.");
}
