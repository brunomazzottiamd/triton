#include <cstddef>
#include <cstdio>
#include <cstdlib>

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

using InTypeA = hipblasLtInt8;
using InTypeB = hipblasLtInt8;
using OutType = hipblasLtInt32;

// Main function:

int main() {
  // Resource acquisition:

  hipStream_t hip_stream;
  CHECK_HIP_ERROR(hipStreamCreate(&hip_stream));

  hipblasLtHandle_t hipblaslt_handle;
  CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&hipblaslt_handle));

  // Buffer sizes:
  const std::size_t size_a = M * K * sizeof(InTypeA);
  const std::size_t size_b = K * N * sizeof(InTypeB);
  const std::size_t size_c = M * N * sizeof(OutType);

  // Host memory:
  void *h_a, *h_b, *h_c;
  CHECK_HIP_ERROR(hipHostMalloc(&h_a, size_a));
  CHECK_HIP_ERROR(hipHostMalloc(&h_b, size_b));
  CHECK_HIP_ERROR(hipHostMalloc(&h_c, size_c));

  // Device memory:
  void *d_a, *d_b, *d_c;
  CHECK_HIP_ERROR(hipMalloc(&d_a, size_a));
  CHECK_HIP_ERROR(hipMalloc(&d_b, size_b));
  CHECK_HIP_ERROR(hipMalloc(&d_c, size_c));

  // Resource cleanup:
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
