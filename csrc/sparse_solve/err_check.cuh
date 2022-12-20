#include <cusparse.h>
#include <cusolverSp.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be a float tensor")

void cusolveSafeCall(cusolverStatus_t err);
void cusparseSafeCall(cusparseStatus_t err);
void cublasSafeCall(cublasStatus_t err);
