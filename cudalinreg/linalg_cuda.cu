/**
 * @file linalg_cuda.cpp
 * @brief A CUDA-accelerated library for calculating simple linear regression.
 *
 * This file provides a single function, run_linear_regression, which uses a CUDA
 * kernel to perform the heavy computation (summation) required for the
 * Ordinary Least Squares (OLS) closed-form solution.
 */
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

/**
 * @brief A macro to wrap CUDA API calls and check for errors.
 * If a CUDA call fails, it prints a detailed error message and exits.
 */
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(result) << " at " <<
            file << ":" << line << " '" << func << "'\n";
        cudaDeviceReset();
        exit(99);
    }
}


/**
 * @brief High-performance kernel using shared memory reduction for linear regression sums.
 * 
 * This kernel replaces the atomicAdd approach with a shared memory reduction pattern.
 * Each block performs a local reduction in fast shared memory, then writes a single
 * partial result to global memory. This dramatically reduces memory contention.
 */
__global__ void closed_form_linalg(double *x_device, double *y_device, double* sums_device, int n)
{
    // 1. DECLARE SHARED MEMORY
    // Dynamically sized shared memory array. Size is set during kernel launch.
    extern __shared__ double s_mem[];
    
    // Partition the shared memory into four arrays, one for each sum we need
    double* s_x = s_mem;
    double* s_y = &s_x[blockDim.x];
    double* s_x2 = &s_y[blockDim.x];
    double* s_xy = &s_x2[blockDim.x];

    // 2. INITIALIZE THREAD VARIABLES
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;

    // Each thread accumulates its results in private registers (fastest memory)
    double my_sum_x = 0.0, my_sum_y = 0.0, my_sum_x2 = 0.0, my_sum_xy = 0.0;

    // 3. GRID-STRIDE LOOP (GLOBAL MEMORY READS)
    // Each thread processes multiple data points if needed
    for (unsigned int j = i; j < n; j += gridDim.x * blockDim.x) {
        double x = x_device[j];
        double y = y_device[j];
        my_sum_x  += x;
        my_sum_y  += y;
        my_sum_x2 += x * x;
        my_sum_xy += x * y;
    }

    // 4. LOAD PARTIAL SUMS INTO SHARED MEMORY
    // Each thread places its accumulated results into the shared memory "workbench"
    s_x[tid] = my_sum_x;
    s_y[tid] = my_sum_y;
    s_x2[tid] = my_sum_x2;
    s_xy[tid] = my_sum_xy;

    // CRITICAL: Wait for all threads in the block to finish loading their data
    __syncthreads();

    // 5. INTRA-BLOCK REDUCTION (SHARED MEMORY TREE REDUCTION)
    // Efficiently combine all partial sums within this block using a tree pattern
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_x[tid] += s_x[tid + s];
            s_y[tid] += s_y[tid + s];
            s_x2[tid] += s_x2[tid + s];
            s_xy[tid] += s_xy[tid + s];
        }
        // Wait for this reduction step to complete before proceeding
        __syncthreads();
    }

    // 6. FINAL WRITE TO GLOBAL MEMORY
    // Only one thread per block writes the final result, eliminating memory contention
    if (tid == 0) {
        sums_device[blockIdx.x * 4 + 0] = s_x[0];
        sums_device[blockIdx.x * 4 + 1] = s_y[0];
        sums_device[blockIdx.x * 4 + 2] = s_x2[0];
        sums_device[blockIdx.x * 4 + 3] = s_xy[0];
    }
}

/**
 * @brief The main public function for this library, callable from other languages.
 * 
 * This function orchestrates the entire linear regression calculation on the GPU.
 * It handles device memory allocation, data transfer, kernel execution, and cleanup.
 * The final calculated slope (m) and intercept (c) are returned via output pointers.
 * 
 * @param x_host Pointer to the host array of x values.
 * @param y_host Pointer to the host array of y values.
 * @param N The number of data points in the x and y arrays.
 * @param m_out Output pointer to store the calculated slope.
 * @param c_out Output pointer to store the calculated intercept.
 */
extern "C" void run_linear_regression(double* x_host, double* y_host, int N, double* m_out, double* c_out) {
    
    // --- 1. Kernel Launch Configuration ---
    // Set the number of threads per block and calculate the number of blocks needed for the grid.
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // --- 2. Device Memory Allocation ---
    // Allocate memory on the GPU for the input arrays and the partial results from each block.
    double *x_device, *y_device, *partial_sums_device;
    checkCudaErrors(cudaMalloc((void**)&x_device, N * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&y_device, N * sizeof(double)));
    // Each block produces 4 partial sums, so we need blocksPerGrid * 4 doubles
    checkCudaErrors(cudaMalloc((void**)&partial_sums_device, blocksPerGrid * 4 * sizeof(double)));

    // --- 3. Host-to-Device Data Transfer ---
    // Copy the input x and y data from the CPU's RAM to the GPU's VRAM.
    checkCudaErrors(cudaMemcpy(x_device, x_host, N * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(y_device, y_host, N * sizeof(double), cudaMemcpyHostToDevice));

    // --- 4. Kernel Execution ---
    // Launch the CUDA kernel with shared memory for the reduction.
    // Shared memory size: 4 arrays * threadsPerBlock * sizeof(double)
    size_t sharedMemSize = threadsPerBlock * 4 * sizeof(double);
    closed_form_linalg<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(x_device, y_device, partial_sums_device, N);
    checkCudaErrors(cudaGetLastError());
    
    // Block the CPU thread until the GPU has finished its work.
    checkCudaErrors(cudaDeviceSynchronize());

    // --- 5. Device-to-Host Data Transfer ---
    // Copy the partial sums from all blocks back to the CPU.
    std::vector<double> partial_sums_host(blocksPerGrid * 4);
    checkCudaErrors(cudaMemcpy(partial_sums_host.data(), partial_sums_device, 
                               blocksPerGrid * 4 * sizeof(double), cudaMemcpyDeviceToHost));

    // --- 6. Final Reduction on CPU ---
    // Sum up the partial results from all blocks to get the final totals.
    double sum_x = 0.0, sum_y = 0.0, sum_x2 = 0.0, sum_xy = 0.0;
    for (int i = 0; i < blocksPerGrid; ++i) {
        sum_x  += partial_sums_host[i * 4 + 0];
        sum_y  += partial_sums_host[i * 4 + 1];
        sum_x2 += partial_sums_host[i * 4 + 2];
        sum_xy += partial_sums_host[i * 4 + 3];
    }

    // Use the OLS closed-form equations to calculate the final slope and intercept.
    double m = (N * sum_xy - sum_x * sum_y) / (N * sum_x2 - sum_x * sum_x);
    double c = (sum_y - m * sum_x) / N;

    // --- 7. Set Output Values ---
    // Dereference the output pointers to write the results back to the caller.
    *m_out = m;
    *c_out = c;

    // --- 8. Cleanup ---
    // Free all memory that was allocated on the GPU device.
    cudaFree(x_device);
    cudaFree(y_device);
    cudaFree(partial_sums_device);
}