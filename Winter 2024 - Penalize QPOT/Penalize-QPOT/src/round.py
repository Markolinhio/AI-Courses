import numpy as np
import numpy.linalg as LA


# Acknowledgement: This code is from APDAGD from Son Pham
# Function to extract matrix and vectors from a flattened input array
def extract(x, m, n):
    """
    Extracts a matrix X of shape (m, n), and two vectors p and q from the input array x.
    """
    X = x[: m * n].reshape(m, n)  # Reshape first m*n elements into an (m, n) matrix
    p = x[m * n : m * n + m]  # Extract the next m elements as vector p
    q = x[m * n + m :]  # Extract the remaining elements as vector q
    return X, p, q


# Function to apply a threshold to avoid numerical instabilities
def threshold_zero(x):
    """
    Ensures all negative values are set to zero and replaces very small values with exact zero.
    """
    x = np.maximum(x, 0)  # Set negative values to zero
    return np.where(
        np.abs(x) > 1e-15, x, np.zeros_like(x)
    )  # Replace very small values with zero


# Function to enforce constraints on a probability distribution
def enforcing_procedure(r, s, pp, ppp):
    """
    Enforces a probability-like constraint to ensure sum(ppp) does not exceed a given threshold.
    """
    alpha = (np.sum(r) - s) / np.sum(pp)
    if alpha < 1:
        pbar = ppp  # Use the current probability distribution if alpha < 1
    else:
        i = -1
        while np.sum(ppp) <= np.sum(r) - s:
            i += 1
            ppp[i] = r[i]  # Adjust values iteratively
        ppp[i] = ppp[i] - (np.sum(ppp) - np.sum(r) + s)  # Final correction
        pbar = ppp
    return pbar


# Function to project an approximate matrix onto a desired row and column sum space
def round_matrix(x, r, c, s):
    """
    Rounds the given matrix X to satisfy row and column sum constraints.
    """
    m = np.shape(r)[0]  # Number of rows
    n = np.shape(c)[0]  # Number of columns
    X, p, q = extract(x, m, n)  # Extract X, p, q from x

    one_m = np.ones(m)  # Vector of ones with size m
    one_n = np.ones(n)  # Vector of ones with size n

    # Compute minimum values for row and column sums
    pp = np.minimum(p, r)
    qp = np.minimum(q, c)

    # Compute scaling factors to enforce constraints
    alpha = min(1.0, (np.sum(r) - s) / np.sum(pp))
    beta = min(1.0, (np.sum(c) - s) / np.sum(qp))

    # Compute adjusted row and column values
    ppp = alpha * pp
    qpp = beta * qp
    X += 1e-12  # Prevent numerical instability

    # Apply enforcing procedure to satisfy constraints
    pbar = enforcing_procedure(r, s, pp, ppp)
    qbar = enforcing_procedure(c, s, qp, qpp)

    # Compute scaling factors to ensure sum constraints are met
    g = np.minimum(one_m, (r - pbar) / X.sum(1))
    h = np.minimum(one_n, (c - qbar) / X.sum(0))
    Xp = np.dot(np.dot(np.diag(g), X), np.diag(h))

    # Compute residual errors for row and column sums
    e1 = (r - pbar) - Xp.sum(1)
    e2 = (c - qbar) - Xp.sum(0)

    # Adjust Xp to obtain final rounded matrix Xbar
    Xbar = Xp + (1.0 / (np.sum(e1))) * np.outer(e1, e2)

    return Xbar, pbar, qbar


# Function to round a matrix using the Sinkhorn projection algorithm
def round_matrix_sinkhorn(X, a, b):
    """
    Projects a given approximate matrix X onto a feasible set with marginals a and b.
    Implementation follows Altschuler et al., 2017.
    """
    one_m = np.ones(X.shape[0])  # Vector of ones with size m (rows)
    one_n = np.ones(X.shape[1])  # Vector of ones with size n (columns)

    # Compute scaling factors for row normalization
    x = a / np.dot(X, one_n)
    x = np.minimum(x, 1)
    F_1 = X.T * x  # Scale columns of X

    # Compute scaling factors for column normalization
    y = b / np.dot(F_1, one_m)
    y = np.minimum(y, 1)
    F_2 = (F_1.T * y).T  # Scale rows of F_1

    # Compute errors in row and column sums
    err_a = a - F_2.T @ one_n
    err_b = b - F_2 @ one_m

    # Compute final projected matrix X_hat
    X_hat = F_2.T + np.outer(err_a, err_b) / LA.norm(err_a, ord=1)

    # Ensure the rounded matrix meets the row and column sum constraints
    assert np.allclose(X_hat.sum(1), a)
    assert np.allclose(X_hat.sum(0), b)

    return X_hat
