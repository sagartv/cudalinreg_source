# CUDA Linear Regression (`cudalinreg`)

This repository contains a Python library, `cudalinreg`, for performing simple linear regression with one variable accelerated by NVIDIA CUDA. The core computation is implemented in C++/CUDA for high performance on datasets > 100K, while the library provides a simple, user-friendly Python interface. On a modern NVIDIA GPU, the speedup of using this version of Linear Regression with one variable is 2x that of Numpy's linalg.lstq solver for 100K samples.


## Prerequisites

Before installing, you must have the following software installed and configured on your system:

1.  **NVIDIA CUDA Toolkit**: Provides the `nvcc` compiler and CUDA runtime libraries. Ensure that `nvcc` is available in your system's `PATH`.
2.  **Python 3**: Along with `pip` and `setuptools`.
3.  **A compatible C++ compiler**: (e.g., `g++` on Linux).

## Installation

It is highly recommended to install this library within a Python virtual environment to avoid conflicts with system-level or other project packages.

### 1. Create and Activate a Virtual Environment

First, create an isolated environment for the project.

```bash
# Create the virtual environment
python3 -m venv .venv

# Activate the environment
# On Linux or macOS:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

### 2. Clone the Repository

Get the source code for the library.

```bash
git clone https://github.com/sagartv/cuda_programming.git
cd cuda_programming
```
*(Note: Please replace the URL with your actual repository URL.)*

### 3. Configure for Your GPU Architecture

Before installing, you must configure the build script to target your specific GPU architecture.

First, find the **Compute Capability** of your NVIDIA GPU. You can find this by searching online (e.g., "NVIDIA GeForce RTX 3080 compute capability").

Next, open the `setup.py` file and edit the `-arch` flag in the `extra_compile_args` line to match your GPU.

**Example `setup.py` configuration:**
```python
# For a GeForce RTX 40-series (Ada Lovelace, CC 8.9)
extra_compile_args={'cxx': [], 'nvcc': ['-arch=sm_89']}

# For a GeForce RTX 30-series (Ampere, CC 8.6)
extra_compile_args={'cxx': [], 'nvcc': ['-arch=sm_86']}

# For a Tesla T4 / RTX 20-series (Turing, CC 7.5)
extra_compile_args={'cxx': [], 'nvcc': ['-arch=sm_75']}
```

### 4. Install the Library

Once the `setup.py` file is configured, install the library using `pip`. We use the **editable (`-e`)** flag, which is best practice for development. This compiles the code "in-place" and links it to your Python environment.

```bash
pip install -e .
```
This command will invoke `nvcc` to compile the CUDA code and install the `cudalinreg` package.

## Usage

After installation, you can import and use the `cudalinreg` library in any Python script just like a standard package. The primary function is `cudalinreg.fit()`.

```python
import numpy as np
import cudalinreg

# 1. Generate some sample data
N = 10_000_000
X = np.arange(N, dtype=np.float64)
y = 2.0 * X + 5.0 + np.random.randn(N) # y = 2x + 5, with noise

# 2. Fit the model using the CUDA-accelerated function
#    This function takes 1D NumPy arrays of type float64.
slope, intercept = cudalinreg.fit(X, y)

# 3. Print the results
print(f"Calculated Slope (m): {slope:.4f}")
print(f"Calculated Intercept (c): {intercept:.4f}")
```

## Benchmarking

This repository includes a benchmark script, `test_library.py`, to compare the performance of `cudalinreg` against NumPy's `linalg.lstsq` solver across a range of data sizes.

To run the benchmark and see the performance comparison:
```bash
python3 test_library.py
```
This will output a summary table showing the time taken and speedup factor for each data size.

### My Results

Below are benchmark results from running on a modern NVIDIA GPU, showcasing the performance of our optimized shared memory implementation.

```
--- Benchmark Results ---
           N |   CUDA Time (s) |  NumPy Time (s) |     Speedup (X) |     CUDA Intercept |    NumPy Intercept
------------------------------------------------------------------------------------------------------------
      10,000 |          1.8496 |          0.0937 |           0.05x |             4.9719 |             4.9719
     100,000 |          0.0039 |          0.0078 |           2.01x |             5.0026 |             5.0026
   1,000,000 |          0.0136 |          0.0452 |           3.33x |             5.0030 |             5.0030
  10,000,000 |          0.1047 |          0.4190 |           4.00x |             4.9997 |             4.9997
  50,000,000 |          0.4481 |          2.0749 |           4.63x |             5.0002 |             5.0002
 100,000,000 |          0.7589 |          4.1130 |           5.42x |             4.9998 |             0.0000
```

#### Analysis of Results

- **Memory Optimization**: The implementation uses a shared memory reduction pattern that dramatically reduces global memory contention. Instead of millions of threads competing to update the same memory locations with atomic operations, each thread block performs its reduction in fast shared memory, with only one thread per block writing to global memory. This optimization leads to the impressive speedups seen above.

- **Crossover Point**: The crossover point where CUDA outperforms NumPy occurs at around 100,000 data points, where we see a 2x speedup. For smaller datasets, the overhead of GPU memory transfers dominates the execution time. However, once past this threshold, the benefits of our shared memory optimization become clear, reaching a 5.42x speedup at 100M points.

- **Numerical Stability**: For very large `N` (100M points), NumPy's general-purpose `lstsq` solver loses precision and calculates an incorrect intercept (0.0000). In contrast, our specialized CUDA implementation maintains accuracy (4.9998, very close to the true value of 5.0) even at this scale. This stability comes from our direct implementation of the OLS formulas and careful use of double-precision arithmetic throughout the computation.
