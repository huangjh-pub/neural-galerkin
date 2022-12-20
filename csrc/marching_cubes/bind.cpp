#include <torch/extension.h>
#include "../common/hashmap_cuda.cuh"

std::vector<torch::Tensor> marching_cubes_sparse_cuda(
        torch::Tensor cube_base,            // (M, 3)
        torch::Tensor cube_sdf             // (M, rx, ry, rz)
);

std::vector<torch::Tensor> dual_mc_sparse_cuda(
        const torch::Tensor& cube_corner_inds,     // (M, 8, 2)   int Tensor
        const torch::Tensor& corner_pos,           // (N, 3)    float Tensor
        const torch::Tensor& corner_value          // (N, )     float Tensor
);

std::vector<torch::Tensor> accumulate_dual_corner_indices(
        const HashLookupData &dual_centers_hash_data, py::dict expand_primal_base, py::dict primal_strides,
        py::dict primal_inds_base);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("marching_cubes_sparse", &marching_cubes_sparse_cuda, "Sparse Structure Marching Cubes (CUDA)");
    m.def("dual_marching_cubes_sparse", &dual_mc_sparse_cuda, "DMC");
    m.def("dual_marching_cubes_indices", &accumulate_dual_corner_indices, "DMC prepare");
}
