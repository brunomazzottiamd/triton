#include <cstddef>
#include <cstdio>
#include <cstdlib>

#include <iostream>
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

static const int M = 4 /*20*/;
static const int N = 5 /*1920*/;
static const int K = 3 /*13312*/;

static const hipDataType HIP_IN_TYPE_A = HIP_R_8I;
static const hipDataType HIP_IN_TYPE_B = HIP_R_8I;
static const hipDataType HIP_OUT_TYPE = HIP_R_32I;

using hipblasLtInTypeA = hipblasLtInt8;
using hipblasLtInTypeB = hipblasLtInt8;
using hipblasLtOutType = hipblasLtInt32;

static const hipblasComputeType_t HIPBLAS_COMPUTE_TYPE = HIPBLAS_COMPUTE_32I;

static const bool TRANS_A = false /*true*/;
static const bool TRANS_B = false;
static const bool TRANS_C = false;

// Generate random input:

template <typename T>
void gen_input(void *h_ptr, int elems, unsigned int seed = 42) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(-5, 5);
  auto ptr = reinterpret_cast<T *>(h_ptr);
  for (int i = 0; i < elems; ++i) {
    const auto value = static_cast<T>(dist(rng));
    ptr[i] = value;
  }
}

// Print matrix:

template <typename T> void print_row_major(void *h_ptr, int rows, int cols) {
  auto ptr = reinterpret_cast<T *>(h_ptr);
  for (int i = 0; i < rows; ++i) {
    const int row_off = i * cols;
    for (int j = 0; j < cols; ++j) {
      std::cout << +(ptr[row_off + j]) << "\t";
    }
    std::cout << "\n";
  }
}

template <typename T> void print_col_major(void *h_ptr, int rows, int cols) {
  auto ptr = reinterpret_cast<T *>(h_ptr);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      const int col_off = j * rows;
      std::cout << +(ptr[col_off + i]) << "\t";
    }
    std::cout << "\n";
  }
}

template <typename T>
void print(const char *desc, void *h_ptr, int rows, int cols, bool trans) {
  std::cout << desc << "(" << rows << ", " << cols << ") = \n";
  if (trans) {
    print_row_major<T>(h_ptr, rows, cols);
  } else {
    print_col_major<T>(h_ptr, rows, cols);
  }
}

template <typename T>
void print_linear(const char *desc, void *h_ptr, int elems) {
  std::cout << desc << "(" << elems << ") = \n";
  auto ptr = reinterpret_cast<T *>(h_ptr);
  for (int i = 0; i < elems; ++i) {
    const T value = ptr[i];
    std::cout << +value << ' ';
  }
  std::cout << "\n";
}

// Other small helper functions:

int lead_dim(int rows, int cols, bool trans) { return trans ? cols : rows; }

hipblasLtOrder_t hipblaslt_order(bool trans) {
  return trans ? HIPBLASLT_ORDER_ROW : HIPBLASLT_ORDER_COL;
}

hipblasOperation_t hipblas_op(bool trans) {
  return trans ? HIPBLAS_OP_T : HIPBLAS_OP_N;
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
  if (elems_a < 30) {
    print_linear<hipblasLtInTypeA>("A", h_a, elems_a);
    print<hipblasLtInTypeA>("A", h_a, M, K, TRANS_A);
  }
  gen_input<hipblasLtInTypeB>(h_b, elems_b, 1947);
  if (elems_b < 30) {
    print_linear<hipblasLtInTypeB>("B", h_b, elems_b);
    print<hipblasLtInTypeB>("B", h_b, K, N, TRANS_B);
  }
  memset(h_c, 0, size_c);
  // print<hipblasLtOutType>("[init] C", h_c, M, N, TRANS_C);

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
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&mat_a, HIP_IN_TYPE_A, M, K,
                                                    lead_dim(M, K, TRANS_A)));
  /*
  const hipblasLtOrder_t order_a = hipblaslt_order(TRANS_A);
  CHECK_HIPBLASLT_ERROR(
      hipblasLtMatrixLayoutSetAttribute(mat_a, HIPBLASLT_MATRIX_LAYOUT_ORDER,
                                        &order_a, sizeof(hipblasLtOrder_t)));
  */
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&mat_b, HIP_IN_TYPE_B, K, N,
                                                    lead_dim(K, N, TRANS_B)));
  /*
  const hipblasLtOrder_t order_b = hipblaslt_order(TRANS_B);
  CHECK_HIPBLASLT_ERROR(
      hipblasLtMatrixLayoutSetAttribute(mat_b, HIPBLASLT_MATRIX_LAYOUT_ORDER,
                                        &order_b, sizeof(hipblasLtOrder_t)));
  */
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&mat_c, HIP_OUT_TYPE, M, N,
                                                    lead_dim(M, N, TRANS_C)));
  /*
  const hipblasLtOrder_t order_c = hipblaslt_order(TRANS_C);
  CHECK_HIPBLASLT_ERROR(
      hipblasLtMatrixLayoutSetAttribute(mat_c, HIPBLASLT_MATRIX_LAYOUT_ORDER,
                                        &order_c, sizeof(hipblasLtOrder_t)));
  */

  // hipBLASLt GEMM descriptor:
  hipblasLtMatmulDesc_t matmul;
  CHECK_HIPBLASLT_ERROR(
      hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_TYPE, HIP_OUT_TYPE));
  const hipblasOperation_t trans_a = hipblas_op(TRANS_A);
  CHECK_HIPBLASLT_ERROR(
      hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA,
                                      &trans_a, sizeof(hipblasOperation_t)));
  const hipblasOperation_t trans_b = hipblas_op(TRANS_B);
  CHECK_HIPBLASLT_ERROR(
      hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB,
                                      &trans_b, sizeof(hipblasOperation_t)));

  // hipBLASLt workspace:
  // TODO: Change workspace size according to the algorithm.
  std::size_t workspace_size = 32 * 1024 * 1024;
  void *d_workspace;
  CHECK_HIP_ERROR(hipMalloc(&d_workspace, workspace_size));

  // Perform hipBLASLt GEMM:
  // D = activation( alpha ⋅ op(A) ⋅ op(B) + beta ⋅ op(C) + bias ) =>
  // D = trans(A) ⋅ B
  // in-place matrix multiplication, i.e. C = D
  hipblasLtOutType alpha = 1;
  hipblasLtOutType beta = 0;
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmul(
      hipblaslt_handle, matmul, reinterpret_cast<void *>(&alpha), d_a, mat_a,
      d_b, mat_b, reinterpret_cast<void *>(&beta), d_c, mat_c, d_c, mat_c,
      nullptr /* algorithm */, d_workspace, workspace_size, hip_stream));

  // Copy from device to host:
  CHECK_HIP_ERROR(
      hipMemcpyAsync(h_c, d_c, size_c, hipMemcpyDeviceToHost, hip_stream));
  hipStreamSynchronize(hip_stream);
  if (elems_c < 30) {
    print_linear<hipblasLtOutType>("[after GEMM] C", h_c, elems_c);
    print<hipblasLtOutType>("[after GEMM] C", h_c, M, N, TRANS_C);
  }

  // Resource cleanup:
  CHECK_HIP_ERROR(hipFree(d_workspace));
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
