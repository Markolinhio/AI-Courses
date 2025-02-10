import numpy
from penalize_qpot import *
from prox import project_onto_simplex
from utils import validate_solution, compute_mu_and_L


def gradient_descent_f(f, X_init, lr=0.1, max_iter=1000, tol=1e-6):
    """
    Optimize f(X) using gradient descent.

    Args:
        f (f): An instance of the f(X) class.
        X_init (np.ndarray): Initial guess for X.
        lr (float): Step size for gradient descent.
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for stopping criterion (norm of gradient).

    Returns:
        X_opt (np.ndarray): Optimized X.
        history (list): List of flattened X arrays during optimization.
        objective_history (list): List of f(X) values.
    """
    X = X_init.copy()
    history = [X.flatten()]  # Track flattened X for visualization
    objective_history = [f(X)]

    for i in range(max_iter):
        grad = f.grad(X)
        X = X - lr * grad  # Gradient descent step
        history.append(X.flatten())  # Append flattened X
        objective_history.append(f(X))  # Track objective value

        # Convergence check
        if np.linalg.norm(grad) < tol:
            print(
                f"Converged after {i + 1} iterations with gradient norm {np.linalg.norm(grad):.6e}"
            )
            break

    return X, history, objective_history


def proximal_gradient_descent(
    F,
    X_init,
    lr=0.1,
    max_iter=1000,
    tol=1e-6,
    decay_rate=0.1,
    step=100,
    eps_tol=1e-4,
    early_stop=False,
    verbose=False,
):
    """
    Proximal gradient descent for F(X) with a simplex projection as the proximal operator.

    Args:
        F: Instance of the F class, representing the smooth part g(x).
        X_init: Initial guess for X (np.ndarray).
        lr: Step size (learning rate).
        max_iter: Maximum number of iterations.
        tol: Tolerance for stopping criterion.
        decay_rate: Rate at which the learning rate decays (default is 10%).
        step: Number of iter before decay
        eps_tol: The threshold of epsilon convergence
        early_stop: If True, stops early if results don't change significantly in the last 5 iterations.
        verbose: If True, prints details of each iteration.

    Returns:
        X_opt: Optimized X.
        history: List of flattened X values for visualization.
        objective_history: List of objective values F(X).
    """
    X = X_init.copy()
    history = [X.flatten()]
    objective_history = [F(X)]
    penalty_history = []
    converge = False
    max_change = 9999
    if verbose:
        print(
            "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
        )
        print(
            f"{'Iteration':<10}|{'Objective Value':<25}|{'f_eta(X)':<12}| {'P(X, alpha)':<22}|{'Sum(X)':<10}|{'Validation':<30}"
        )
        print(
            "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
        )
    for k in range(max_iter):
        # Gradient descent step
        grad = F.grad(X)
        Y = X - lr * grad

        # Proximal step: Project onto simplex
        X_new = project_onto_simplex(Y, F.s)

        # Track history
        history.append(X_new.flatten())
        current_obj = F(X_new)
        objective_history.append(F(X_new))
        current_P_alpha = F.P_alpha(X_new)  # Compute P(X, alpha)
        penalty_history.append(current_P_alpha)
        # Verbose output
        if verbose:
            current_f_eta = F.f_eta(X_new)  # Compute f_eta(X)
            # Check projection status
            sum_x = np.sum(X_new)
            min_x = np.min(X_new)

            # Validation result
            validation_result = "PASS"
            try:
                validate_solution(F, X_new, eps_tol)
                converge = True
            except AssertionError as e:
                validation_result = f"{str(e)}"

            # Print debugging details in tabular format
            print(
                f"{k + 1:<10}|{current_obj:<25.6f}| {current_f_eta:<11.6f}| {current_P_alpha:<22.6f}|{np.sum(X_new):<10.4f}|{validation_result:<30}"
            )

        # Early stopping: If the change in the last 5 objectives is very small
        if early_stop and len(objective_history) > 5:
            max_change = np.max(np.abs(np.diff(objective_history[-5:])))
        if converge:
            if early_stop and max_change < tol:
                print(
                    f"Early stopping activated at iteration {k + 1}: "
                    f"Max change in last 5 objectives is {max_change:.6e}."
                )
                break

        # Update learning rate with decay
        if k % step == 0:
            lr *= 1 - decay_rate

        X = X_new

    return X, history, objective_history, penalty_history


def nesterov_accelerated_gradient(
    F,
    X_init,
    mu=None,
    L=None,
    max_iter=1000,
    tol=1e-6,
    step=50,
    decay_rate=0.5,
    eps_tol=1e-4,
    early_stop=False,
    verbose=False,
    line_search=True,
    beta=0.3,  # Reduced beta for stricter line search
    L_init=1.0,
):
    """
    Nesterov's Accelerated Gradient (NAG) method with adjustments for stability and constraints.

    Args:
        F: Instance of the F class (objective function).
        X_init: Initial guess for X (np.ndarray).
        mu: Strong convexity parameter. If None, computed automatically.
        L: Smoothness parameter. If None, computed automatically or by line search.
        max_iter: Maximum number of iterations.
        tol: Tolerance for stopping criterion.
        step: Number of iterations before decaying the learning rate.
        decay_rate: Decay rate for learning rate.
        eps_tol: Epsilon tolerance for validation.
        early_stop: If True, stops early if results don't change significantly in the last 5 iterations.
        verbose: If True, prints details of each iteration.
        line_search: If True, uses backtracking line search to estimate L.
        beta: Reduction factor for line search (default 0.3).
        L_init: Initial guess for L in line search.

    Returns:
        X_opt: Optimized X.
        history: List of flattened X values for visualization.
        objective_history: List of objective values F(X).
        penalty_history: List of penalty values P(X, alpha).
    """
    # Automatically compute mu and L if not provided
    if mu is None or L is None:
        mu, L = compute_mu_and_L(F)

    # Initialize variables
    Y = X_init.copy()
    X = X_init.copy()
    A = 1e-4
    L = L if L else L_init
    history = [X.flatten()]
    objective_history = [F(X)]
    penalty_history = []
    converge = False
    recent_objectives = []
    A_history = [A]
    max_change = 9999

    if verbose:
        print(
            "-------------------------------------------------------------------------------------------------------------------------------------------------------"
        )
        print(
            f"{'Iteration':<10}|{'Objective Value':<25}|{'f_eta(X)':<12}| {'P(X, alpha)':<22}|{'Sum(X)':<10}|{'Validation':<30}"
        )
        print(
            "-------------------------------------------------------------------------------------------------------------------------------------------------------"
        )

    for k in range(max_iter):
        # Backtracking line search to estimate L
        if line_search:
            L = L_init  # Reset L to initial guess for each iteration
            while True:
                GRAD_Y = F.grad(Y)
                X_temp = project_onto_simplex(Y - (1 / L) * GRAD_Y, F.s)
                lhs = F(X_temp)
                rhs = F(Y) - (1 / (2 * L)) * np.linalg.norm(GRAD_Y, ord="fro") ** 2
                if lhs <= rhs and validate_solution(F, X_temp, eps_tol):
                    break
                L *= 1 / beta  # Increase L

        # Step 3: Compute new A
        A_new = A / (1 - np.sqrt(mu / L))

        # Step 4: Update new X
        GRAD_Y = F.grad(Y)
        X_new = project_onto_simplex(Y - (1 / L) * GRAD_Y, F.s)

        # Step 5: Update new Y
        decay = (1 - decay_rate) if k % step == 0 and not line_search else 1
        Y_new = X_new + decay * (1 - np.sqrt(mu / L)) / (1 + np.sqrt(mu / L)) * (
            X_new - X
        )

        # Track history
        history.append(X_new.flatten())
        current_obj = F(X_new)
        objective_history.append(current_obj)
        current_P_alpha = F.P_alpha(X_new)
        penalty_history.append(current_P_alpha)
        A_history.append(A_new)

        # Verbose output
        if verbose:
            current_f_eta = F.f_eta(X_new)
            sum_X = np.sum(X_new)

            # Validation result
            validation_result = "PASS"
            try:
                validate_solution(F, X_new, eps_tol)
                converge = True
            except AssertionError as e:
                validation_result = f"{str(e)}"

            # Print debugging details in tabular format
            print(
                f"{k + 1:<10}|{current_obj:<25.6f}| {current_f_eta:<11.6f}| {current_P_alpha:<22.6f}|{sum_X:<10.4f}|{validation_result:<30}"
            )

        # Early stopping: If the change in the last 5 objectives is very small
        if early_stop and len(objective_history) > 5:
            max_change = np.max(np.abs(np.diff(objective_history[-5:])))
        if converge:
            if early_stop and max_change < tol:
                print(
                    f"Early stopping activated at iteration {k + 1}: "
                    f"Max change in last 5 objectives is {max_change:.6e}."
                )
                break

        # Update variables for next iteration
        A = A_new
        X = X_new
        Y = Y_new

    return X_new, history, objective_history, penalty_history, A_history
