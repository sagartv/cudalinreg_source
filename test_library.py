import numpy as np
import cudalinreg
import time
import sys

def run_comparison(N):
    """
    Runs a single comparison for a given data size N, ensuring both
    methods use the same data.
    """
    # 1. Prepare data
    X_data = np.arange(N, dtype=np.float64)
    y_data = 2.0 * X_data + 5.0 + np.random.randn(N)

    # --- Time the CUDA implementation ---
    start_time_cuda = time.time()
    slope_cuda, intercept_cuda = cudalinreg.fit(X_data, y_data)
    end_time_cuda = time.time()
    cuda_duration = end_time_cuda - start_time_cuda

    # --- Time the NumPy implementation (using the linalg closed-form solution) ---
    start_time_numpy = time.time()

    # Using the general-purpose linalg.lstsq solver.
    # This represents the direct "closed-form" solution in NumPy's linalg module.
    # NOTE: This method is known to have numerical stability issues with
    # large, uncentered data, which is why the intercept is inaccurate for large N.
    A = np.vstack([X_data, np.ones(len(X_data))]).T
    
    # np.linalg.lstsq returns a tuple; the solution is the first element
    slope_numpy, intercept_numpy = np.linalg.lstsq(A, y_data, rcond=None)[0]

    end_time_numpy = time.time()
    numpy_duration = end_time_numpy - start_time_numpy

    return {
        "N": N,
        "slope_cuda": slope_cuda,
        "intercept_cuda": intercept_cuda,
        "time_cuda": cuda_duration,
        "slope_numpy": slope_numpy,
        "intercept_numpy": intercept_numpy,
        "time_numpy": numpy_duration,
    }


def main():
    """
    A benchmark script to compare the performance of cudalinreg with NumPy
    across various data sizes.
    """
    # A range of N values to test scalability
    N_values = [
        10_000, 
        100_000, 
        1_000_000, 
        10_000_000, 
        50_000_000,
        100_000_000
    ]

    print("--- Performance Benchmark: custom CUDA vs. NumPy ---")
    print("Running tests for various data sizes (N). This may take a few moments...")
    
    results = []
    for N in N_values:
        # Use a carriage return to show progress on a single line
        sys.stdout.write(f"\rRunning test for N = {N:<12}...")
        sys.stdout.flush()
        results.append(run_comparison(N))
    
    print("\rBenchmark complete.                               ") # Clear the line

    # --- Print the Results ---
    print("\n--- Benchmark Results ---")
    header = f"{'N':>12} | {'CUDA Time (s)':>15} | {'NumPy Time (s)':>15} | {'Speedup (X)':>15} | {'CUDA Intercept':>18} | {'NumPy Intercept':>18}"
    print(header)
    print("-" * len(header))

    for res in results:
        # Avoid division by zero if a run is extremely fast
        if res['time_cuda'] > 0:
            speedup = res['time_numpy'] / res['time_cuda']
        else:
            speedup = float('inf')

        print(
            f"{res['N']:>12,} | "
            f"{res['time_cuda']:>15.4f} | "
            f"{res['time_numpy']:>15.4f} | "
            f"{speedup:>14.2f}x | "
            f"{res['intercept_cuda']:>18.4f} | "
            f"{res['intercept_numpy']:>18.4f}"
        )
    
    print("\n--- Analysis ---")
    print("The 'Speedup' column compares your specialized CUDA kernel against NumPy's general-purpose `linalg.lstsq` solver.")
    print("This comparison highlights the significant performance advantage a specialized algorithm can have over a")
    print("general-purpose one, especially when leveraging GPU parallelism.")
    
    print("\nNote on NumPy's Inaccurate Intercept:")
    print("You will notice `lstsq` computes an incorrect intercept for large N. This is not a bug in NumPy,")
    print("but a fundamental issue of numerical instability (catastrophic cancellation) when a general")
    print("linear algebra solver operates on large, uncentered data. Achieving both speed and accuracy")
    print("often requires specialized numerical techniques, like those in your library or by pre-processing the data.")


if __name__ == '__main__':
    main() 