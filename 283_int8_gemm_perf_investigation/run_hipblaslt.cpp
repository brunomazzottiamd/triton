#include <cstddef>
#include <cstdio>
#include <cstdlib>

#include <iostream>
#include <random>
#include <vector>

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
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

static const hipblasOperation_t TRANS_A = HIPBLAS_OP_N /*HIPBLAS_OP_T*/;
static const hipblasOperation_t TRANS_B = HIPBLAS_OP_N;
static const hipblasOperation_t TRANS_C = HIPBLAS_OP_N;

static const hipDataType HIP_IN_TYPE_A = HIP_R_8I;
static const hipDataType HIP_IN_TYPE_B = HIP_R_8I;
static const hipDataType HIP_OUT_TYPE_C = HIP_R_8I /*HIP_R_32I*/;

using hipblasLtInTypeA = hipblasLtInt8;
using hipblasLtInTypeB = hipblasLtInt8;
using hipblasLtOutTypeC = hipblasLtInt8 /*hipblasLtInt32*/;

static const hipblasComputeType_t HIPBLAS_COMPUTE_TYPE = HIPBLAS_COMPUTE_32I;

static const int M = 4 /*20*/;
static const int N = 5 /*1920*/;
static const int K = 3 /*13312*/;

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
void print(const char *desc, void *h_ptr, int rows, int cols,
           hipblasOperation_t trans) {
  std::cout << desc << "(" << rows << ", " << cols << ") = \n";
  if (trans == HIPBLAS_OP_T) {
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

// Main function:

int main() {
  //////// Resource acquisition:

  // HIP stream:
  hipStream_t hip_stream;
  CHECK_HIP_ERROR(hipStreamCreate(&hip_stream));

  // hipBLASLT handle:
  hipblasLtHandle_t hipblaslt_handle;
  CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&hipblaslt_handle));

  // Buffer sizes:
  const int elems_a = M * K;
  const std::size_t size_a = elems_a * sizeof(hipblasLtInTypeA);
  const int elems_b = K * N;
  const std::size_t size_b = elems_b * sizeof(hipblasLtInTypeB);
  const int elems_c = M * N;
  const std::size_t size_c = elems_c * sizeof(hipblasLtOutTypeC);

  // Allocate host memory:
  void *h_a, *h_b, *h_c;
  CHECK_HIP_ERROR(hipHostMalloc(&h_a, size_a));
  CHECK_HIP_ERROR(hipHostMalloc(&h_b, size_b));
  CHECK_HIP_ERROR(hipHostMalloc(&h_c, size_c));

  // Allocate device memory:
  void *d_a, *d_b, *d_c;
  CHECK_HIP_ERROR(hipMalloc(&d_a, size_a));
  CHECK_HIP_ERROR(hipMalloc(&d_b, size_b));
  CHECK_HIP_ERROR(hipMalloc(&d_c, size_c));

  //////// Fill host memory:

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

  //////// Copy from host to device:

  CHECK_HIP_ERROR(
      hipMemcpyAsync(d_a, h_a, size_a, hipMemcpyHostToDevice, hip_stream));
  CHECK_HIP_ERROR(
      hipMemcpyAsync(d_b, h_b, size_b, hipMemcpyHostToDevice, hip_stream));

  //////// hipBLASLt GEMM with C++ API:

  hipblaslt_ext::Gemm gemm(hipblaslt_handle, TRANS_A, TRANS_B, HIP_IN_TYPE_A,
                           HIP_IN_TYPE_B, HIP_OUT_TYPE_C, HIP_OUT_TYPE_C,
                           HIPBLAS_COMPUTE_TYPE);
  hipblaslt_ext::GemmEpilogueV2 epilogue;
  hipblaslt_ext::GemmInputsV2 inputs;
  inputs.setA(d_a);
  inputs.setB(d_b);
  inputs.setC(d_c);
  inputs.setD(d_c);
  const hipblasLtOutTypeC alpha = 1;
  const hipblasLtOutTypeC beta = 0;
  inputs.setAlpha(&alpha);
  inputs.setBeta(&beta);
  gemm.setProblem(M, N, K, 1, epilogue, inputs);

  // hipBLASLt workspace:
  // TODO: Change workspace size according to the algorithm.
  std::size_t workspace_size = 32 * 1024 * 1024;
  void *d_workspace;
  CHECK_HIP_ERROR(hipMalloc(&d_workspace, workspace_size));
  hipblaslt_ext::GemmPreferenceV2 pref;
  pref.setMaxWorkspaceBytes(workspace_size);

  // Search for hipBLASLt algorithm:
  const int request_solutions = 1;
  std::vector<hipblasLtMatmulHeuristicResult_t> heuristic_result;
  CHECK_HIPBLASLT_ERROR(
      gemm.algoGetHeuristic(request_solutions, pref, heuristic_result));

  if (heuristic_result.empty()) {
    std::cerr << "No valid solution found!\n";
    return 1;
  }

  // Perform hipBLASLt GEMM:
  // D = activation( alpha ⋅ op(A) ⋅ op(B) + beta ⋅ op(C) + bias )
  // When
  // * alpha = 1
  // * beta = 0
  // * bias = activation = none
  // * C = D (in-place matrix multiplication)
  // then we have:
  // C = op(A) ⋅ op(B)
  CHECK_HIPBLASLT_ERROR(gemm.initialize(heuristic_result[0].algo, d_workspace));
  CHECK_HIPBLASLT_ERROR(gemm.run(hip_stream));

  //////// Copy from device to host:

  CHECK_HIP_ERROR(
      hipMemcpyAsync(h_c, d_c, size_c, hipMemcpyDeviceToHost, hip_stream));
  hipStreamSynchronize(hip_stream);
  if (elems_c < 30) {
    print_linear<hipblasLtOutTypeC>("[after GEMM] C", h_c, elems_c);
    print<hipblasLtOutTypeC>("[after GEMM] C", h_c, M, N, TRANS_C);
  }

  //////// Resource cleanup:

  CHECK_HIP_ERROR(hipFree(d_workspace));
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
