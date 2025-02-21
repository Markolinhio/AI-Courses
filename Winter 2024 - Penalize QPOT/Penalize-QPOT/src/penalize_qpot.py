import numpy as np


class f:
    """
    Primal objective function for f_eta(X) = <C, X> + eta * ||X||_2^2.
    """

    def __init__(self, C, eta):
        """
        Args:
            C (np.ndarray): Cost matrix.
            eta (float): Regularization parameter.
        """
        self.C = C
        self.eta = eta

    def __call__(self, X):
        """
        Compute f_eta(X) = <C, X> + eta * ||X||_2^2.

        Args:
            X (np.ndarray): Input matrix.

        Returns:
            float: Value of the function.
        """
        # Compute <C, X> + eta * ||X||_2^2
        inner_product = np.sum(self.C * X)
        frobenius_norm_squared = np.linalg.norm(X, ord="fro") ** 2
        return inner_product + self.eta * frobenius_norm_squared

    def grad(self, X):
        """
        Compute the gradient of f_eta(X) with respect to X.

        Args:
            X (np.ndarray): Input matrix.

        Returns:
            np.ndarray: Gradient matrix of the same shape as X.
        """
        return self.C + 2 * self.eta * X

    def hess(self, X):
        """
        Compute the Hessian of f_eta(X) with respect to X.
        The Hessian is a diagonal operator scaled by 2 * eta.

        Args:
            X (np.ndarray): Input matrix (not used in this case as the Hessian is constant).

        Returns:
            np.ndarray: A diagonal matrix representing the Hessian (2 * eta * I).
        """
        # Hessian is 2 * eta times the identity matrix, for simplicity return diagonal operator
        shape = X.shape
        return 2 * self.eta * np.eye(np.prod(shape)).reshape(*shape, *shape)


class P:
    """
    Regularization function P(X, alpha) with gradient computation.
    """

    def __init__(self, r, c, alpha):
        """
        Args:
            r (np.ndarray): Row constraint vector of shape (n,).
            c (np.ndarray): Column constraint vector of shape (m,).
            alpha (float): Regularization parameter.
        """
        self.r = r
        self.c = c
        self.alpha = alpha

    def __call__(self, X):
        """
        Compute P(X, alpha).

        Args:
            X (np.ndarray): Input matrix.

        Returns:
            float: Value of P(X, alpha).
        """
        # Row and column sums
        row_sums = np.sum(X, axis=1)  # X * 1_n
        col_sums = np.sum(X, axis=0)  # X^T * 1_n

        # Compute min(0, r_i - row_sums_i)^2
        row_term = np.minimum(0, self.r - row_sums) ** 2

        # Compute min(0, c_i - col_sums_i)^2
        col_term = np.minimum(0, self.c - col_sums) ** 2

        # Combine terms with alpha
        return self.alpha * (np.sum(row_term) + np.sum(col_term))

    def grad(self, X):
        """
        Compute the gradient of P(X, alpha) using indicator functions.

        Args:
            X (np.ndarray): Input matrix.

        Returns:
            np.ndarray: Gradient matrix of the same shape as X.
        """
        n, n = X.shape

        # Step 1: Compute row and column sums
        row_sums = np.sum(X, axis=1)  # Sum across rows (shape: n,)
        col_sums = np.sum(X, axis=0)  # Sum across columns (shape: m,)

        # Step 2: Identify active constraints for rows and columns
        row_active = (
            self.r - row_sums
        ) < 0  # Boolean mask for row violations (shape: n,)
        col_active = (
            self.c - col_sums
        ) < 0  # Boolean mask for column violations (shape: m,)

        # Initialize gradient matrix
        gradient = np.zeros_like(X, dtype=np.float64)

        # Step 3: Compute gradient contributions from rows
        row_grad_coeffs = (
            -2 * self.alpha * (self.r - row_sums) * row_active
        )  # Gradient coefficients for rows (shape: n,)
        # Apply gradient contribution to all elements in the row
        for i in range(n):
            if row_active[i]:
                gradient[i, :] += row_grad_coeffs[i]

        # Step 4: Compute gradient contributions from columns
        col_grad_coeffs = (
            -2 * self.alpha * (self.c - col_sums) * col_active
        )  # Gradient coefficients for columns (shape: m,)
        # Add contribution to the column
        for j in range(n):
            if col_active[j]:
                gradient[:, j] += col_grad_coeffs[j]

        # Step 5: Combine the row and column gradient contributions
        return gradient


class F:
    """
    Combined objective function F(X) = f_eta(X) + P(X, alpha), with gradient computation.
    """

    def __init__(self, C, eta, r, c, alpha, s, eps_tol=0):
        """
        Args:
            C (np.ndarray): Cost matrix.
            eta (float): Regularization parameter for f_eta.
            r (np.ndarray): Row constraint vector of shape (n,).
            c (np.ndarray): Column constraint vector of shape (m,).
            alpha (float): Regularization parameter for P.
            s (float): Sum of transported mass
        """
        self.C = C
        self.eta = eta
        self.alpha = alpha
        self.eps_tol = eps_tol
        self.r = r + self.eps_tol
        self.c = c + self.eps_tol
        self.s = s
        self.f_eta = f(self.C, self.eta)
        self.P_alpha = P(self.r, self.c, self.alpha)

    def __call__(self, X):
        """
        Compute F(X) = f_eta(X) + P(X, alpha).

        Args:
            X (np.ndarray): Input matrix.

        Returns:
            float: Value of F(X).
        """
        return self.f_eta(X) + self.P_alpha(X)

    def grad(self, X):
        """
        Compute the gradient of F(X) = grad(f_eta(X)) + grad(P(X, alpha)).

        Args:
            X (np.ndarray): Input matrix.

        Returns:
            np.ndarray: Gradient matrix of the same shape as X.
        """
        grad_f_eta = self.f_eta.grad(X)
        grad_P_alpha = self.P_alpha.grad(X)
        return grad_f_eta + grad_P_alpha
