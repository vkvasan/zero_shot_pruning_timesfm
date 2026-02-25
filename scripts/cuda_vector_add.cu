#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

static void check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(err));
    std::exit(1);
  }
}

int main() {
  constexpr int n = 1 << 20;
  constexpr size_t bytes = n * sizeof(float);

  std::vector<float> h_a(n), h_b(n), h_c(n);
  for (int i = 0; i < n; ++i) {
    h_a[i] = 0.5f * i;
    h_b[i] = 2.0f * i;
  }

  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  check(cudaMalloc(&d_a, bytes), "cudaMalloc(d_a)");
  check(cudaMalloc(&d_b, bytes), "cudaMalloc(d_b)");
  check(cudaMalloc(&d_c, bytes), "cudaMalloc(d_c)");

  check(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D a");
  check(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D b");

  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
  check(cudaGetLastError(), "kernel launch");
  check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

  check(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H c");

  double max_abs_err = 0.0;
  for (int i = 0; i < n; ++i) {
    double expected = static_cast<double>(h_a[i]) + static_cast<double>(h_b[i]);
    max_abs_err = std::max(max_abs_err, std::abs(static_cast<double>(h_c[i]) - expected));
  }

  std::printf("vector_add OK, n=%d, max_abs_err=%.6g\n", n, max_abs_err);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  return 0;
}

