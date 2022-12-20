#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cusparse.h>
#include <cusolverSp.h>
#include <time.h>
#include "err_check.cuh"

cusolverSpHandle_t solver_handle = nullptr;

void init_cusolver_handle() {
    cusolveSafeCall(cusolverSpCreate(&solver_handle));
}

// CuSolver only supports CSR matrix.
torch::Tensor solve_cusparse(const torch::Tensor& Ap, const torch::Tensor& Aj, const torch::Tensor& Ax, const torch::Tensor& b,
                             float tol) {
    CHECK_CONTIGUOUS(Ap); CHECK_CUDA(Ap); CHECK_IS_INT(Ap);
    CHECK_CONTIGUOUS(Aj); CHECK_CUDA(Aj); CHECK_IS_INT(Aj);
    CHECK_CONTIGUOUS(Ax); CHECK_CUDA(Ax); CHECK_IS_FLOAT(Ax);
    CHECK_CONTIGUOUS(b); CHECK_CUDA(b); CHECK_IS_FLOAT(b);

    cusparseMatDescr_t descr;
    cusparseSafeCall(cusparseCreateMatDescr(&descr));
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);    // can be symmetric, triangular, etc.
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    int n = b.size(0);
    int nnz = Ax.size(0);
    torch::Tensor x = torch::empty({n}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    float* x_data = x.data_ptr<float>();

    cudaStreamSynchronize(at::cuda::getCurrentCUDAStream());

    int singularity;
    cusolveSafeCall(cusolverSpScsrlsvchol(solver_handle, n, nnz, descr, Ax.data_ptr<float>(),
                    Ap.data_ptr<int>(), Aj.data_ptr<int>(), b.data_ptr<float>(),
                                          tol, 3, x_data, &singularity));

    cusparseSafeCall(cusparseDestroyMatDescr(descr));

    return x;
}
