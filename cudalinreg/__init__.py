import ctypes
import numpy as np
import os
import glob

# --- Load the Shared Library ---
# This part is a bit tricky. We need to find the compiled .so file.
# When installed with pip, it's in the package directory.
try:
    # Find the compiled .so file within the package directory
    lib_path = glob.glob(os.path.join(os.path.dirname(__file__), '_linalg*.so'))[0]
    cuda_lib = ctypes.CDLL(lib_path)
except IndexError:
    raise OSError("Could not find the compiled CUDA library. Please make sure the package is installed correctly.")


# --- Define the Function Signature ---
# Tell ctypes what the argument and return types are for our C++ function.
_run_linear_regression_c = cuda_lib.run_linear_regression
_run_linear_regression_c.argtypes = [
    ctypes.POINTER(ctypes.c_double), # x_host
    ctypes.POINTER(ctypes.c_double), # y_host
    ctypes.c_int,                    # N
    ctypes.POINTER(ctypes.c_double), # m_out
    ctypes.POINTER(ctypes.c_double)  # c_out
]


def fit(X, y):
    """
    Fits a linear regression model to the provided data using a CUDA-accelerated backend.

    This function is designed to feel like a standard library function from
    Scikit-learn or similar libraries.

    Args:
        X (np.ndarray): A 1D NumPy array of the independent variable (features).
                        The data should be of type np.float64.
        y (np.ndarray): A 1D NumPy array of the dependent variable (target).
                        The data should be of type np.float64.

    Returns:
        A tuple containing the calculated slope (m) and intercept (c).
    """
    # --- Input Validation ---
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("Input X and y must be NumPy arrays.")
    if X.ndim != 1 or y.ndim != 1:
        raise ValueError("Input arrays X and y must be 1-dimensional.")
    if len(X) != len(y):
        raise ValueError("Input arrays X and y must have the same length.")
    if X.dtype != np.float64 or y.dtype != np.float64:
        raise TypeError("Input arrays X and y must be of dtype np.float64.")

    N = len(X)

    # --- Prepare Output Variables ---
    # Create variables that ctypes can pass by reference (as pointers)
    m_out = ctypes.c_double()
    c_out = ctypes.c_double()

    # --- Call the C++ CUDA Function ---
    _run_linear_regression_c(
        X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        N,
        ctypes.byref(m_out),
        ctypes.byref(c_out)
    )

    return m_out.value, c_out.value 