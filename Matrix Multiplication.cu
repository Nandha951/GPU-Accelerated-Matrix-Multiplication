#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono> // For timing

// Helper function to check CUDA errors
void checkCudaError(cudaError_t error) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
        exit(1);
    }
}

// Kernel for matrix multiplication
__global__ void matrixMultiply(float* A, float* B, float* C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row index for C
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Column index for C

    if (row < m && col < k) { // Bounds check
        float sum = 0;
        for (int i = 0; i < n; i++) {
            sum += A[row * n + i] * B[i * k + col];
        }
        C[row * k + col] = sum;
    }
}

// CPU function for matrix multiplication
void matrixMultiplyCPU(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int m, int n, int k) {
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < k; ++col) {
            float sum = 0;
            for (int i = 0; i < n; ++i) {
                sum += A[row * n + i] * B[i * k + col];
            }
            C[row * k + col] = sum;
        }
    }
}

int main() {
    int m = 1024; // Rows in A and C
    int n = 1024; // Columns in A and rows in B
    int k = 1024; // Columns in B and C

    // Host memory
    std::vector<float> h_A(m * n);
    std::vector<float> h_B(n * k);
    std::vector<float> h_C(m * k);

    // Initialize matrices (example values - SIMPLE VALUES FOR DEBUGGING)
    for (int i = 0; i < m * n; ++i) {
        h_A[i] = 1.0f; // All 1s for A
    }
    for (int i = 0; i < n * k; ++i) {
        h_B[i] = 1.0f; // All 1s for B
    }


    // Device memory
    float* d_A;
    float* d_B;
    float* d_C;

    // Allocate memory on the device
    checkCudaError(cudaMalloc(&d_A, m * n * sizeof(float)));
    checkCudaError(cudaMalloc(&d_B, n * k * sizeof(float)));
    checkCudaError(cudaMalloc(&d_C, m * k * sizeof(float)));

    // Copy data from host to device
    checkCudaError(cudaMemcpy(d_A, h_A.data(), m * n * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_B, h_B.data(), n * k * sizeof(float), cudaMemcpyHostToDevice));

    // Kernel configuration (adjust block size as needed)
    int blockSize = 16; // Example: 16x16 threads per block
    dim3 blockDim(blockSize, blockSize);
    dim3 gridDim((k + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y);

    // Time the GPU calculation
    auto start_gpu = std::chrono::high_resolution_clock::now();

    // Launch the kernel
    matrixMultiply<<<gridDim, blockDim>>>(d_A, d_B, d_C, m, n, k);
    checkCudaError(cudaDeviceSynchronize()); // Wait for kernel to finish
    auto end_gpu = std::chrono::high_resolution_clock::now();
    auto duration_gpu = std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu);

    // Copy result back to host
    checkCudaError(cudaMemcpy(h_C.data(), d_C, m * k * sizeof(float), cudaMemcpyDeviceToHost));

    // CPU Calculation and Timing
    std::vector<float> h_C_cpu(m * k); // Separate vector for CPU results
    auto start_cpu = std::chrono::high_resolution_clock::now();
    matrixMultiplyCPU(h_A, h_B, h_C_cpu, m, n, k);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto duration_cpu = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu);

    // Verify results (compare GPU and CPU results)
    bool results_match = true;
    float tolerance = 1e-5; // Adjust tolerance as needed
    for (int i = 0; i < m * k; ++i) {
        if (std::abs(h_C[i] - h_C_cpu[i]) > tolerance) {
            results_match = false;
            std::cerr << "Mismatch at element " << i << ": GPU=" << h_C[i] << ", CPU=" << h_C_cpu[i] << std::endl;
            //break; // Keep checking all elements to see all mismatches
        }
    }

    if (results_match) {
        std::cout << "Results match!" << std::endl;
    } else {
        std::cout << "Results do not match." << std::endl;
    }

    // Print timings
    std::cout << "GPU Time: " << duration_gpu.count() << " ms" << std::endl;
    std::cout << "CPU Time: " << duration_cpu.count() << " ms" << std::endl;

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

// Compile with: nvcc -o matrix_multiplication matrix_multiplication.cu
// Run with: ./matrix_multiplication
// Example output:
// Results match!
// GPU Time: 27 ms
// CPU Time: 26011 ms