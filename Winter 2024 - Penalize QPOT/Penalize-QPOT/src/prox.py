import numpy as np
from utils import matrix_to_vector, vector_to_matrix


def simplex_projection_1d(y):
    """
    Projects a 1D vector y onto the probability simplex.

    Args:
        y (np.ndarray): Input vector of shape (D,).

    Returns:
        np.ndarray: Vector projected onto the simplex.
    """
    sorted_y = np.sort(y)[::-1]
    cumulative_sum = np.cumsum(sorted_y)
    rho = np.where(sorted_y + (1 - cumulative_sum) / (np.arange(len(y)) + 1) > 0)[0][-1]
    lambda_val = (1 - cumulative_sum[rho]) / (rho + 1)
    return np.maximum(y + lambda_val, 0)


def project_onto_simplex(Y, s, verbose=False):
    """Projects Y onto the simplex to enforce non-negativity and sum to size of s."""
    Y_flat = matrix_to_vector(Y)
    if verbose:
        print("Input dimension:", Y.shape)
        print("Intemediary dimension:", Y_flat.shape)
        print("Sorted dimension: ", np.sort(Y_flat / s)[::-1].shape)
        print("Cumulative sum:", np.cumsum(np.sort(Y_flat / s)[::-1]))
        print(
            "rho:",
            np.where(
                np.sort(Y_flat / s)[::-1]
                + (1 - np.cumsum(np.sort(Y_flat / s)[::-1]))
                / (np.arange(len(Y_flat)) + 1)
                > 0
            )[0][-1],
        )
    Y_projected = simplex_projection_1d(Y_flat / s)
    return vector_to_matrix(Y_projected * s, *Y.shape)
