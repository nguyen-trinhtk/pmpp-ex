/* matrix-add.cu
PMPP Exercise 3.9.1 (pg 67)
a. Write host stub function
b. Kernel 1: each thread compute one output entry
c. Kernel 2: each thread compute one output row
d. Kernel 3: each thread compute one output column
*/ 

#include <random>
#include <iostream>
#include <cmath>
#include <cstdint>
#include <cstring> 
#include <cstdlib>
#include <algorithm>

#define CEIL_DIV(n, m) ((n + m - 1)/m) // ceil(n / m)

// ===================== Constants ===================== 
constexpr double SINGLE_EPS = 1e-6; // Single precision tolerance?

// ===================== Host helper functions ===================== 
// float random_finite_float(std::mt19937& gen) {
//     constexpr uint32_t exponent_mask = 0x7F800000u;
//     uint32_t bits;
//     do { bits = static_cast<uint32_t>(gen()); }
//     while ((bits & exponent_mask) == exponent_mask); // Reject Inf and NaN
//     float f;
//     std::memcpy(&f, &bits, sizeof f);                // type-pun without bit_cast
//     return f;
// }

float random_float_range(std::mt19937& gen, float start, float end) {
    std::uniform_real_distribution<float> dist(start, end);  // uniform in [start, end)
    return dist(gen);
}

void random_fill(float* matrix, size_t n) {
    std::random_device rd;
    std::mt19937 gen(rd()); // Mersenne Twister
    for (int i = 0; i < n * n; i++) {
        matrix[i] = random_float_range(gen, -1e3f, 1e3f);    
    }
    return;
}

void host_add_matrix(double* C, float* A, float* B, size_t N) {
    for (int i = 0; i < N * N; i++) {
        C[i] = static_cast<double>(A[i]) + static_cast<double>(B[i]);
    } return;
}

void check_error(float* ans, double* ref, size_t n) {
    int failed = 0; 
    double max_rre = 0; 
    for (int i = 0; i < n * n; i++) {
        double rre = (std::abs(static_cast<double>(ans[i]) - ref[i])) / (std::abs(ref[i]));
        if (rre > SINGLE_EPS) {
            std::cout << "[ERROR] RRE =" << rre <<  "> EPS, at entry (" 
                << (i % n) << ", " << (i / n) << ")" << std::endl;
            failed++; 
        }
        max_rre = std::max(rre, max_rre);
    }
    std::cout << "Max RRE: " << max_rre << std::endl;
    if (failed == 0) {
        std::cout << "[PASS] All entries checked :)" << std::endl;
    }
}

// ===================== CUDA kernels ===================== 
__global__
void naiveMatAddKernel(float* dC, float* dA, float* dB, size_t N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // col
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row

    if (x < N && y < N) {
        int i = y * N + x;
        dC[i] = dA[i] + dB[i]; // each thread compute one output
    }
}

// Slow (nearby thread access strided memory locs)
__global__
void rowMatAddKernel(float* dC, float* dA, float* dB, size_t N) {
    int r = blockIdx.x * blockDim.x + threadIdx.x; 
    if (r < N) {
        for (int i = r * N; i < r * N + N; i++) {
            dC[i] = dA[i] + dB[i];
        }
    }
}

// Col kernel is faster than row kernel because its coalesced
__global__
void colMatAddKernel(float* dC, float* dA, float* dB, size_t N) {
    int c = blockIdx.x * blockDim.x + threadIdx.x; 
    if (c < N) {
        for (int i = c; i < N * N; i+=N) {
            dC[i] = dA[i] + dB[i];
        }
    }
}

// ===================== Main ===================== 

// Stub
void ker_add_matrix(float* C, float* A, float* B, size_t N) {
    float *d_A, *d_B, *d_C; 
    int size = N * N;
    // TODO: on subsequent exercises, use cudaMallocManaged instead
    cudaMalloc((void**) &d_A, size * sizeof(float));
    cudaMalloc((void**) &d_B, size * sizeof(float));
    cudaMalloc((void**) &d_C, size * sizeof(float));

    cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size * sizeof(float), cudaMemcpyHostToDevice);

    // CUDA Event
    cudaEvent_t start, stop; 
    cudaEventCreate(&start); 
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // // b. Each thread compute one output
    // size_t block_size = 32;
    // size_t grid_size = CEIL_DIV(N, block_size);
    // dim3 grid(grid_size, grid_size, 1);
    // dim3 block(block_size, block_size, 1);
    // naiveMatAddKernel<<<grid, block>>>(d_C, d_A, d_B, N);

    // c. Each thread compute one row
    // size_t block_size = 256;
    // size_t grid_size = CEIL_DIV(N * N, block_size);
    // dim3 grid(grid_size, 1, 1);
    // dim3 block(block_size, 1, 1);
    // rowMatAddKernel<<<grid, block>>>(d_C, d_A, d_B, N);

    // // d. Each thread compute one column
    size_t block_size = 256;
    size_t grid_size = CEIL_DIV(N * N, block_size);
    dim3 grid(grid_size, 1, 1);
    dim3 block(block_size, 1, 1);
    colMatAddKernel<<<grid, block>>>(d_C, d_A, d_B, N);

    cudaEventRecord(stop);
    cudaDeviceSynchronize(); // wait event stop fully

    // Time elapsed
    float ms = 0.0f; 
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    std::cout << "Kernel time elapsed: " << ms << " ms" << std::endl;

    // Copy back to host and free device buffers
    cudaMemcpy(C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A); 
    cudaFree(d_B);
    cudaFree(d_C);
    return;
}

// Main
int main() {
    // std::cout << "C++ version: " << __cplusplus << '\n';

    // Size
    size_t N = 1 << 10; 
    size_t N_square = N * N;

    // Allocate host data
    float* A = static_cast<float*>(malloc(N_square * sizeof(float)));
    float* B = static_cast<float*>(malloc(N_square * sizeof(float)));
    float* C = static_cast<float*>(malloc(N_square * sizeof(float)));
    double* C_host = static_cast<double*>(malloc(N_square * sizeof(double)));
    
    // Fill host data randomly
    random_fill(A, N); 
    random_fill(B, N);

    // Stub function
    ker_add_matrix(C, A, B, N); // C = A + B

    // Check error drift vs CPU 64-bit baseline
    host_add_matrix(C_host, A, B, N);
    check_error(C, C_host, N); // numerical error

    // Free
    free(A); A = nullptr;
    free(B); B = nullptr;
    free(C); C = nullptr;
    free(C_host); C_host = nullptr;

    return 0;
}