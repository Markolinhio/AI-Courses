import numpy as np


def matrix_to_vector(matrix):
    """
    Convert a 2D matrix to a 1D vector, preserving row-column relationships.

    Args:
        matrix (np.ndarray): A 2D numpy array.

    Returns:
        vector (np.ndarray): A 1D numpy array containing all elements of the matrix.
    """
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise ValueError("Input must be a 2D numpy array.")

    return matrix.flatten(order="C")  # Flatten in row-major order (default)


def vector_to_matrix(vector, rows, cols):
    """
    Convert a 1D vector back to a 2D matrix with specified dimensions.

    Args:
        vector (np.ndarray): A 1D numpy array.
        rows (int): The number of rows for the output matrix.
        cols (int): The number of columns for the output matrix.

    Returns:
        matrix (np.ndarray): A 2D numpy array with the specified dimensions.
    """
    if not isinstance(vector, np.ndarray) or vector.ndim != 1:
        raise ValueError("Input must be a 1D numpy array.")

    if len(vector) != rows * cols:
        raise ValueError(
            "The size of the vector does not match the specified dimensions."
        )

    return vector.reshape((rows, cols), order="C")  # Reshape in row-major order


def calculate_alpha(r, c, s, epsilon, eta, C):
    """
    Calculate the appropriate alpha value based on the theorem.

    Args:
        r: Row constraint vector (np.ndarray).
        c: Column constraint vector (np.ndarray).
        s: Scalar sum constraint.
        epsilon: Desired epsilon convergence value.
        eta: Regularization parameter.
        C: Cost matrix (np.ndarray of size n x n).

    Returns:
        alpha: Calculated penalty coefficient alpha.
    """
    n = C.shape[0]  # Dimension of the cost matrix

    # Calculate h
    h = np.sqrt(2 * n + 1) / 2

    # Calculate z
    z = 0  # Assuming z = 0 as stated in the theorem

    # Calculate f_eta(X)
    norm_C_inf = np.max(np.abs(C))  # ||C||_\infty
    f_eta_bound = s * norm_C_inf + eta * s**2

    # Calculate \zeta
    zeta_r = np.min(r)
    zeta_c = np.min(c)
    zeta_min = 1 / n * min(np.sum(r), np.sum(c) - s)
    zeta = min(zeta_r, zeta_c, zeta_min)

    # Ensure zeta <= \min g_i(X)
    min_g_i_X = zeta

    # Calculate \bar{alpha}
    alpha_bar = (z - (-f_eta_bound)) / min_g_i_X

    # Calculate alpha
    alpha = (alpha_bar * h) / epsilon

    return alpha


def compute_mu_and_L(F_obj):
    """
    Computes the strong convexity parameter (mu) and smoothness parameter (L)
    for the combined objective F_{\eta, \alpha}(X) based on Lemma 6 and Lemma 7.

    Args:
        F_obj: Instance of the F class containing problem definitions.

    Returns:
        mu: Strong convexity parameter.
        L: Smoothness parameter.
    """
    # Compute mu based on Lemma 6
    mu = F_obj.eta  # Strong convexity parameter is eta

    # Compute L based on Lemma 7
    n = F_obj.C.shape[0]  # Dimension of the cost matrix
    alpha = F_obj.alpha  # Penalty parameter
    L = n * alpha  # Smoothness parameter is O(n * alpha)

    return mu, L


def validate_solution(F_obj, X_opt, epsilon_tolerance):
    """
    Validates the solution X_opt based on the constraints of F_obj.

    Args:
        F_obj: Instance of the F class containing problem definitions (C, eta, r, c, alpha, s).
        X_opt: Optimized solution matrix.
        epsilon_tolerance: Allowed tolerance for constraint satisfaction.

    Returns:
        None. Prints validation results for each constraint.
    """
    # Test 1: Check if all elements in X_opt are non-negative
    assert np.all(X_opt >= 0), "FAIL: Some elements in X_opt are negative."

    # Test 2: Check if the sum of elements in X_opt equals s (within tolerance)
    assert np.isclose(
        np.sum(X_opt), F_obj.s, atol=epsilon_tolerance
    ), f"FAIL: Sum of elements in X_opt ({np.sum(X_opt)}) does not equal s ({F_obj.s}) within tolerance."

    # Test 3 & 4: Check if row and column sums are within their respective constraints with tolerance
    row_sums = np.sum(X_opt, axis=1)
    col_sums = np.sum(X_opt, axis=0)

    row_check = np.all(row_sums <= F_obj.r + epsilon_tolerance)
    col_check = np.all(col_sums <= F_obj.c + epsilon_tolerance)

    row_max_diff = str(np.round(np.max(row_sums - (F_obj.r + epsilon_tolerance)), 5))
    row_max_idx = str(np.argmax(row_sums - (F_obj.r + epsilon_tolerance)))
    col_max_diff = str(np.round(np.max(col_sums - (F_obj.c + epsilon_tolerance)), 5))
    col_max_idx = str(np.argmax(col_sums - (F_obj.c + epsilon_tolerance)))

    assert row_check and col_check, (
        "FAIL: "
        f"{'Row failed with max difference: ' + row_max_diff  if not row_check else 'Row passed,'} "
        f"{'Column failed with max difference: ' + col_max_diff if not col_check else 'Col passed'}"
    )
