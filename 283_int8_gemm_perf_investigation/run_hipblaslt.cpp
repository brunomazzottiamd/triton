#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <random>

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

// Error handling:

#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(error)                                                 \
  if (error != hipSuccess) {                                                   \
    std::fprintf(stderr, "Hip error: '%s'(%d) at %s:%d\n",                     \
                 hipGetErrorString(error), error, __FILE__, __LINE__);         \
    std::exit(EXIT_FAILURE);                                                   \
  }
#endif

#ifndef CHECK_HIPBLASLT_ERROR
#define CHECK_HIPBLASLT_ERROR(error)                                           \
  if (error != HIPBLAS_STATUS_SUCCESS) {                                       \
    std::fprintf(stderr, "hipBLASLt error(Err=%d) at %s:%d\n", error,          \
                 __FILE__, __LINE__);                                          \
    std::exit(EXIT_FAILURE);                                                   \
  }
#endif

// GEMM specification:

static const int M = 20;
static const int N = 1920;
static const int K = 13312;

static const hipDataType HIP_IN_TYPE_A = HIP_R_8I;
static const hipDataType HIP_IN_TYPE_B = HIP_R_8I;
static const hipDataType HIP_OUT_TYPE = HIP_R_32I;

using hipblasLtInTypeA = hipblasLtInt8;
using hipblasLtInTypeB = hipblasLtInt8;
using hipblasLtOutType = hipblasLtInt32;

static const hipblasComputeType_t HIPBLAS_COMPUTE_TYPE = HIPBLAS_COMPUTE_32I;

// Generate random input:

template <typename T>
void gen_input(void *h_ptr, int elems, unsigned int seed = 42) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(-5, 5);
  for (int i = 0; i < elems; ++i) {
    (reinterpret_cast<T *>(h_ptr))[i] = static_cast<T>(dist(rng));
  }
}

// Main function:

int main() {
  // Resource acquisition:

  hipStream_t hip_stream;
  CHECK_HIP_ERROR(hipStreamCreate(&hip_stream));

  hipblasLtHandle_t hipblaslt_handle;
  CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&hipblaslt_handle));

  // Buffer sizes:
  const int elems_a = M * K;
  const std::size_t size_a = elems_a * sizeof(hipblasLtInTypeA);
  const int elems_b = K * N;
  const std::size_t size_b = elems_b * sizeof(hipblasLtInTypeB);
  const int elems_c = M * N;
  const std::size_t size_c = elems_c * sizeof(hipblasLtOutType);

  // Allocate host memory:
  void *h_a, *h_b, *h_c;
  CHECK_HIP_ERROR(hipHostMalloc(&h_a, size_a));
  CHECK_HIP_ERROR(hipHostMalloc(&h_b, size_b));
  CHECK_HIP_ERROR(hipHostMalloc(&h_c, size_c));

  // Fill host memory:
  gen_input<hipblasLtInTypeA>(h_a, elems_a, 1983);
  gen_input<hipblasLtInTypeB>(h_b, elems_b, 1947);

  // Allocate device memory:
  void *d_a, *d_b, *d_c;
  CHECK_HIP_ERROR(hipMalloc(&d_a, size_a));
  CHECK_HIP_ERROR(hipMalloc(&d_b, size_b));
  CHECK_HIP_ERROR(hipMalloc(&d_c, size_c));

  // Copy from host to device:
  CHECK_HIP_ERROR(
      hipMemcpyAsync(d_a, h_a, size_a, hipMemcpyHostToDevice, hip_stream));
  CHECK_HIP_ERROR(
      hipMemcpyAsync(d_b, h_b, size_b, hipMemcpyHostToDevice, hip_stream));

  // hipBLASLt matrix layouts:
  hipblasLtMatrixLayout_t mat_a, mat_b, mat_c;
  // row major
  CHECK_HIPBLASLT_ERROR(
      hipblasLtMatrixLayoutCreate(&mat_a, HIP_IN_TYPE_A, M, K, K));
  // column major
  CHECK_HIPBLASLT_ERROR(
      hipblasLtMatrixLayoutCreate(&mat_b, HIP_IN_TYPE_B, K, N, K));
  // column major (pay attention to this matrix, Triton seems to use row major
  // layout)
  CHECK_HIPBLASLT_ERROR(
      hipblasLtMatrixLayoutCreate(&mat_c, HIP_OUT_TYPE, M, N, M));

  // hipBLASLt GEMM descriptor:
  hipblasLtMatmulDesc_t matmul;
  CHECK_HIPBLASLT_ERROR(
      hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_TYPE, HIP_OUT_TYPE));
  const hipblasOperation_t trans_a = HIPBLAS_OP_T; // transposed
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
      matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(std::int32_t)));
  const hipblasOperation_t trans_b = HIPBLAS_OP_N; // non-transposed
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
      matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(std::int32_t)));

  // Resource cleanup:
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmul));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(mat_c));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(mat_b));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(mat_a));
  CHECK_HIP_ERROR(hipFree(d_c));
  CHECK_HIP_ERROR(hipFree(d_b));
  CHECK_HIP_ERROR(hipFree(d_a));
  CHECK_HIP_ERROR(hipFree(h_c));
  CHECK_HIP_ERROR(hipFree(h_b));
  CHECK_HIP_ERROR(hipFree(h_a));
  CHECK_HIPBLASLT_ERROR(hipblasLtDestroy(hipblaslt_handle));
  CHECK_HIP_ERROR(hipStreamDestroy(hip_stream));

  return 0;
}
