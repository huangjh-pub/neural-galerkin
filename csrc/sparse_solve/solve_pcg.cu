#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cusparse.h>
#include <cusolverSp.h>
#include <time.h>
#include "err_check.cuh"

__global__ void apply_jacobi(const float *a, const float *b, float *res, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        res[i] = a[i] * b[i];
    }
}

std::pair<torch::Tensor, int> solve_pcg_diag(
        const torch::Tensor& Ap, const torch::Tensor& Aj, const torch::Tensor& Ax, const torch::Tensor& b,
        const torch::Tensor& inv_diag_A,
        const float tol, const int max_iter, const bool res_fix) {

    CHECK_CONTIGUOUS(Ap); CHECK_CUDA(Ap); CHECK_IS_INT(Ap);
    CHECK_CONTIGUOUS(Aj); CHECK_CUDA(Aj); CHECK_IS_INT(Aj);
    CHECK_CONTIGUOUS(Ax); CHECK_CUDA(Ax); CHECK_IS_FLOAT(Ax);
    CHECK_CONTIGUOUS(b); CHECK_CUDA(b); CHECK_IS_FLOAT(b);
    CHECK_CONTIGUOUS(inv_diag_A); CHECK_CUDA(inv_diag_A); CHECK_IS_FLOAT(inv_diag_A);

    int N = b.size(0);
    int nz = Ax.size(0);
    int sqrt_n = (int) std::ceil(std::sqrt((double) N));

    float b_norm = torch::linalg_norm(b).item<float>();
    float atol = tol * b_norm;

    torch::Tensor x_tensor = torch::zeros({N}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor d_p_tensor = torch::zeros({N}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor d_z_tensor = torch::zeros({N}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor d_r_tensor = torch::clone(b);
    torch::Tensor Ax_tensor = torch::zeros({N}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    float* d_x = x_tensor.data_ptr<float>();
    float* d_p = d_p_tensor.data_ptr<float>();
    float* d_z = d_z_tensor.data_ptr<float>();
    float* d_r = d_r_tensor.data_ptr<float>();
    float* d_Ax = Ax_tensor.data_ptr<float>();
    float* d_b = b.data_ptr<float>();
    const float* d_inv_diag_A = inv_diag_A.data_ptr<float>();

    cusparseHandle_t cusparseHandle = at::cuda::getCurrentCUDASparseHandle();
    cublasHandle_t cublasHandle = at::cuda::getCurrentCUDABlasHandle();

    cusparseSpMatDescr_t matA = NULL;
    cusparseSafeCall(cusparseCreateCsr(&matA, N, N, nz, Ap.data_ptr<int>(), Aj.data_ptr<int>(), Ax.data_ptr<float>(),
                                       CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                       CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    cusparseDnVecDescr_t vecx = NULL;
    cusparseSafeCall(cusparseCreateDnVec(&vecx, N, d_x, CUDA_R_32F));
    cusparseDnVecDescr_t vecp = NULL;
    cusparseSafeCall(cusparseCreateDnVec(&vecp, N, d_p, CUDA_R_32F));
    cusparseDnVecDescr_t vecAx = NULL;
    cusparseSafeCall(cusparseCreateDnVec(&vecAx, N, d_Ax, CUDA_R_32F));

    float one = 1.0;
    float neg_one = -1.0;
    float zero = 0.0;
    size_t bufferSize = 0;
    cusparseSafeCall(cusparseSpMV_bufferSize(
            cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, vecx,
            &zero, vecAx, CUDA_R_32F, CUSPARSE_MV_ALG_DEFAULT, &bufferSize));
    torch::Tensor cusparse_buffer_tensor = torch::zeros({(int)bufferSize}, torch::dtype(torch::kByte).device(torch::kCUDA));
    void *buffer = cusparse_buffer_tensor.data_ptr<unsigned char>();

    cusparseSafeCall(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  &one, matA, vecx, &zero, vecAx, CUDA_R_32F,
                                  CUSPARSE_MV_ALG_DEFAULT, buffer));
    cublasSafeCall(cublasSaxpy(cublasHandle, N, &neg_one, d_Ax, 1, d_r, 1));

    int iters = 0;
    float rho = 0.0;
    float rho1;

    while (max_iter < 0 || iters < max_iter) {
        {
            dim3 dimBlock = dim3(256);
            dim3 dimGrid = dim3((N + dimBlock.x - 1) / dimBlock.x);
            apply_jacobi<<<dimGrid, dimBlock>>>(d_inv_diag_A, d_r, d_z, N);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }

        rho1 = rho;

        cublasSafeCall(cublasSdot(cublasHandle, N, d_r, 1, d_z, 1, &rho));

        if (iters == 0) {
            cublasSafeCall(cublasScopy(cublasHandle, N, d_z, 1, d_p, 1));
        } else {
            float betap = rho / rho1;
            cublasSafeCall(cublasSscal(cublasHandle, N, &betap, d_p, 1));
            cublasSafeCall(cublasSaxpy(cublasHandle, N, &one, d_z, 1, d_p, 1));
        }

        cusparseSafeCall(cusparseSpMV(
                cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, vecp,
                &zero, vecAx, CUDA_R_32F, CUSPARSE_MV_ALG_DEFAULT, buffer));

        float alpha, neg_alpha, dot;
        cublasSafeCall(cublasSdot(cublasHandle, N, d_p, 1, d_Ax, 1, &dot));
        alpha = rho / dot;
        cublasSafeCall(cublasSaxpy(cublasHandle, N, &alpha, d_p, 1, d_x, 1));

        if ((iters + 1) % sqrt_n == 0 && res_fix) {
            cusparseSafeCall(cusparseSpMV(
                    cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, vecx,
                    &zero, vecAx, CUDA_R_32F, CUSPARSE_MV_ALG_DEFAULT, buffer));
            cublasSafeCall(cublasScopy(cublasHandle, N, d_b, 1, d_r, 1));
            cublasSafeCall(cublasSaxpy(cublasHandle, N, &neg_one, d_Ax, 1, d_r, 1));
        } else {
            neg_alpha = -alpha;
            cublasSafeCall(cublasSaxpy(cublasHandle, N, &neg_alpha, d_Ax, 1, d_r, 1));
        }

        iters++;

        float resid;
        cublasSafeCall(cublasSnrm2(cublasHandle, N, d_r, 1, &resid));
        if (resid <= atol) {
            break;
        }

    }

    cusparseSafeCall(cusparseDestroySpMat(matA));
    cusparseSafeCall(cusparseDestroyDnVec(vecx));
    cusparseSafeCall(cusparseDestroyDnVec(vecAx));
    cusparseSafeCall(cusparseDestroyDnVec(vecp));

    return std::make_pair(x_tensor, iters);
}
