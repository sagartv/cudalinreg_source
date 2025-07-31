# Technical Implementation Details

This document provides a deep dive into the parallel computation and memory optimization strategies used in the `cudalinreg` library.

## Linear Regression Algorithm

### Mathematical Foundation

The library implements Ordinary Least Squares (OLS) linear regression using the closed-form solution:

For a line `y = mx + c`, given N points (x₁,y₁), ..., (xₙ,yₙ), the optimal parameters are:

```
m = (N∑xy - ∑x∑y) / (N∑x² - (∑x)²)
c = (∑y - m∑x) / N
```

where ∑ denotes summation from i=1 to N.

### Why Parallel Computing Helps

The computation requires four summations: ∑x, ∑y, ∑x², and ∑xy. These summations:
1. Are independent of each other
2. Can be computed in any order (associative)
3. Can be split into partial sums and later combined (distributive)

These properties make the algorithm highly parallelizable.

## CUDA Implementation

### Memory Hierarchy Utilization

Our implementation leverages CUDA's memory hierarchy for optimal performance:

1. **Register Memory (Fastest)**
   - Each thread maintains private accumulators for partial sums
   - Variables: `my_sum_x`, `my_sum_y`, `my_sum_x2`, `my_sum_xy`
   - Used in the grid-stride loop for initial accumulation

2. **Shared Memory (Fast)**
   - Size: `threadsPerBlock * 4 * sizeof(double)` bytes
   - Partitioned into four arrays for the different sums
   - Used for intra-block reduction
   - Access latency: ~100x faster than global memory

3. **Global Memory (Slow)**
   - Input arrays: `x_device`, `y_device`
   - Output array: `partial_sums_device`
   - Only used for initial data loading and final results

### Parallel Reduction Algorithm

The implementation uses a two-level reduction strategy:

#### Level 1: Thread-Block Level Reduction
```cpp
// 1. Each thread accumulates multiple elements (grid-stride loop)
for (unsigned int j = i; j < n; j += gridDim.x * blockDim.x) {
    double x = x_device[j];
    double y = y_device[j];
    my_sum_x  += x;
    my_sum_y  += y;
    my_sum_x2 += x * x;
    my_sum_xy += x * y;
}

// 2. Load private sums into shared memory
s_x[tid]  = my_sum_x;
s_y[tid]  = my_sum_y;
s_x2[tid] = my_sum_x2;
s_xy[tid] = my_sum_xy;
__syncthreads();  // Ensure all threads have written

// 3. Tree-based reduction in shared memory
for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
        s_x[tid]  += s_x[tid + s];
        s_y[tid]  += s_y[tid + s];
        s_x2[tid] += s_x2[tid + s];
        s_xy[tid] += s_xy[tid + s];
    }
    __syncthreads();  // Ensure reduction step is complete
}
```

#### Level 2: Final CPU Reduction
```cpp
// On CPU: Sum partial results from all blocks
double sum_x = 0.0, sum_y = 0.0, sum_x2 = 0.0, sum_xy = 0.0;
for (int i = 0; i < blocksPerGrid; ++i) {
    sum_x  += partial_sums_host[i * 4 + 0];
    sum_y  += partial_sums_host[i * 4 + 1];
    sum_x2 += partial_sums_host[i * 4 + 2];
    sum_xy += partial_sums_host[i * 4 + 3];
}
```

### Memory Access Optimization

1. **Coalesced Global Memory Access**
   - Adjacent threads access adjacent memory locations
   - Achieved through the grid-stride loop pattern
   - Maximizes memory bandwidth utilization

2. **Shared Memory Bank Conflicts**
   - Avoided by using sequential indices within warps
   - Each thread accesses its own bank in shared memory

3. **Memory Transfer Minimization**
   - Only two transfers: input data (host→device) and partial sums (device→host)
   - Partial sums size: only `blocksPerGrid * 4` doubles

### Performance Characteristics

1. **Computational Complexity**
   - Time Complexity: O(N/P) where P is number of parallel threads
   - Space Complexity: O(N) for input + O(B) for partial sums (B = number of blocks)

2. **Memory Bandwidth**
   - Reads: 2N doubles (x and y arrays)
   - Writes: 4B doubles (partial sums, B = number of blocks)
   - Total Transfer: (2N + 4B) * sizeof(double) bytes

3. **Synchronization Points**
   - Two `__syncthreads()` per block
   - One global synchronization (`cudaDeviceSynchronize()`)

## Comparison with Previous Approaches

### Atomic Operations Approach

The previous implementation used `atomicAdd` to update global counters:
```cpp
atomicAdd(&sums_device[0], x);
atomicAdd(&sums_device[1], y);
atomicAdd(&sums_device[2], x * x);
atomicAdd(&sums_device[3], x * y);
```

Issues with this approach:
1. **Memory Contention**: All threads compete for 4 memory locations
2. **Serialization**: Atomic operations effectively serialize access
3. **Cache Thrashing**: Continuous updates to same locations

### Shared Memory Benefits

The new implementation solves these issues:
1. **Reduced Contention**: Only one write per block to global memory
2. **Better Parallelism**: No serialization within blocks
3. **Cache Efficiency**: Shared memory acts as a programmer-managed cache

## Future Optimization Possibilities

1. **Warp-Level Primitives**
   - Use `__shfl_down_sync()` for faster warp-level reduction
   - Could eliminate some shared memory operations

2. **Multi-GPU Support**
   - Distribute data across multiple GPUs
   - Combine results using MPI or similar

3. **Stream Processing**
   - Process data in chunks for larger-than-memory datasets
   - Use CUDA streams for concurrent execution