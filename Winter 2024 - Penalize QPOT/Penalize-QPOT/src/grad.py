import numpy
from penalize_qpot import *
from prox import project_onto_simplex
from utils import validate_solution, compute_mu_and_L, measure_sparsity


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
    F_obj,
    X_init,
    lr=0.1,
    max_iter=1000,
    tol=1e-6,
    decay_rate=0.1,
    step=100,
    early_stop=None,
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
        early_stop: If True, stops early if results don't change significantly in the last 5 iterations.
        verbose: If True, prints details of each iteration.

    Returns:
        X_opt: Optimized X.
        history: List of flattened X values for visualization.
        objective_history: List of objective values F(X).
    """
    X = X_init.copy()
    history = [X.flatten()]
    objective_history = [F_obj(X)]
    penalty_history = []
    converge = False
    max_change = 9999
    if verbose:
        print(
            "------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
        )
        print(
            f"{'Iteration':<10}|{'Objective Value':<25}|{'f_eta(X)':<12}| {'P(X, alpha)':<22}|{'Sum(X)':<10}|{'Sparsity':<10}|{'Convergence':<30}"
        )
        print(
            "------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
        )
    for k in range(max_iter):
        # Gradient descent step
        grad = F_obj.grad(X)
        Y = X - lr * grad

        # Proximal step: Project onto simplex
        X_new = project_onto_simplex(Y, F_obj.s)

        # Track history
        history.append(X_new.flatten())
        current_obj = F_obj(X_new)
        objective_history.append(F_obj(X_new))
        current_P_alpha = F_obj.P_alpha(X_new)  # Compute P(X, alpha)
        penalty_history.append(current_P_alpha)

        # Validation and check convergence result
        validation_result = "PASS"
        try:
            validate_solution(F_obj, X_new, 0)
            if measure_sparsity(X_new) >= 0.6:
                converge = True
            else:
                converge = False
                validation_result = "FAIL: Sparsity too low"
        except AssertionError as e:
            validation_result = f"{str(e)}"
            converge = False

        # Verbose output
        if verbose:
            current_f_eta = F_obj.f_eta(X_new)  # Compute f_eta(X)
            sum_X = np.sum(X_new)
            convergence_status = "PASS" if converge else validation_result
            sparsity = measure_sparsity(X_new)
            # Print debugging details in tabular format
            print(
                f"{k + 1:<10}|{current_obj:<25.6f}| {current_f_eta:<11.6f}| {current_P_alpha:<22.6f}|{sum_X:<10.4f}|{sparsity:<10.4f}|{convergence_status:<30}"
            )

        # Early stopping: If the change in the last 5 objectives is very small
        if isinstance(early_stop, int):
            if early_stop > 0 and len(objective_history) > early_stop:
                max_change = np.max(
                    np.abs(np.diff(objective_history[-1 * early_stop :]))
                )
            if converge:
                if (
                    early_stop > 0
                    and len(objective_history) > early_stop
                    and max_change < tol
                ):
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
    F_obj,
    X_init,
    mu=None,
    L=None,
    max_iter=1000,
    tol=1e-6,
    step=50,
    decay_rate=0.5,
    early_stop=None,
    sparsity_threshold=0.6,  # Sparsity stopping criterion
    sparsity_tol=1e-10,  # Threshold for counting sparse elements
    verbose=False,
    line_search=True,
    beta=0.3,  # Reduced beta for stricter line search
    L_init=1.0,
    patience=100,  # Number of iterations to wait after last improvement
):
    """
    Nesterov's Accelerated Gradient (NAG) method with robust sparsity-based best solution tracking.

    Includes:
    - Best-solution tracking based on sparsity with convergence guarantee.
    - Ensures that once a converged best solution is found, we do not continue needlessly.
    - Within the patience window, updates to a better sparse solution if found.

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
        early_stop: If True, stops early if results don't change significantly in last `patience` iterations.
        sparsity_threshold: If sparsity exceeds this value, include in convergence check.
        sparsity_tol: Value threshold to consider elements as zero.
        verbose: If True, prints details of each iteration.
        line_search: If True, uses backtracking line search to estimate L.
        beta: Reduction factor for line search (default 0.3).
        L_init: Initial guess for L in line search.
        patience: Number of iterations to wait before stopping after last improvement.

    Returns:
        X_best: Best solution found (highest sparsity with convergence).
        history: List of flattened X values for visualization.
        objective_history: List of objective values.
        penalty_history: List of penalty values.
    """

    # Automatically compute mu and L if not provided
    if mu is None or L is None:
        mu, L = compute_mu_and_L(F_obj)

    # Initialize variables
    Y = X_init.copy()
    X = X_init.copy()
    A = 1e-4
    L = L if L else L_init
    history = [X.flatten()]
    objective_history = [F_obj(X)]
    penalty_history = []
    sparsity_history = [measure_sparsity(X, threshold=sparsity_tol)]
    A_history = [A]

    # Best solution tracking (based on highest sparsity with convergence)
    X_best = X.copy()
    best_sparsity = sparsity_history[-1] if sparsity_history else 0
    best_iteration = 0  # Track the best iteration
    patience_counter = 0
    converge = False
    best_converged_found = False  # Ensure we track a truly converged best solution

    if verbose:
        print(
            "------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
        )
        print(
            f"{'Iteration':<10}|{'Objective Value':<25}|{'f_eta(X)':<12}| {'P(X, alpha)':<22}|{'Sum(X)':<10}|{'Sparsity':<10}|{'Convergence':<30}"
        )
        print(
            "------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
        )

    for k in range(max_iter):
        # Backtracking line search to estimate L
        if line_search:
            L = L_init  # Reset L to initial guess for each iteration
            while True:
                GRAD_Y = F_obj.grad(Y)
                X_temp = project_onto_simplex(Y - (1 / L) * GRAD_Y, F_obj.s)
                lhs = F_obj(X_temp)
                rhs = F_obj(Y) - (1 / (2 * L)) * np.linalg.norm(GRAD_Y, ord="fro") ** 2
                if lhs <= rhs and validate_solution(F_obj, X_temp, 0):
                    break
                L *= 1 / beta  # Increase L

        # Step 3: Compute new A
        A_new = A / (1 - np.sqrt(mu / L))

        # Step 4: Update new X
        GRAD_Y = F_obj.grad(Y)
        X_new = project_onto_simplex(Y - (1 / L) * GRAD_Y, F_obj.s)

        # Step 5: Update new Y
        decay = (1 - decay_rate) if k % step == 0 and not line_search else 1
        Y_new = X_new + decay * (1 - np.sqrt(mu / L)) / (1 + np.sqrt(mu / L)) * (
            X_new - X
        )

        # Compute sparsity
        sparsity = measure_sparsity(X_new, threshold=sparsity_tol)
        sparsity_history.append(sparsity)

        # Track history
        history.append(X_new.flatten())
        current_obj = F_obj(X_new)
        objective_history.append(current_obj)
        current_P_alpha = F_obj.P_alpha(X_new)
        penalty_history.append(current_P_alpha)
        A_history.append(A_new)

        # Validate and check convergence
        try:
            validate_solution(F_obj, X_new, 0)
            if sparsity >= sparsity_threshold:
                converge = True
            else:
                validation_result = "FAIL: Sparisty too low"
        except AssertionError as e:
            converge = False
            validation_result = f"{str(e)}"

        # Verbose output
        if verbose:
            current_f_eta = F_obj.f_eta(X_new)
            sum_X = np.sum(X_new)
            convergence_status = "PASS" if converge else validation_result

            print(
                f"{k + 1:<10}|{current_obj:<25.6f}| {current_f_eta:<11.6f}| {current_P_alpha:<22.6f}|{sum_X:<10.4f}|{sparsity:<10.4f}|{convergence_status:<30}"
            )

        # Best solution tracking (update only if it's converged and more sparse)
        # print(converge, sparsity > best_sparsity)
        if converge and sparsity > best_sparsity:
            best_sparsity = sparsity
            X_best = X_new.copy()
            best_iteration = k + 1  # Track the best iteration (1-based index)
            patience_counter = 0  # Reset patience
            best_converged_found = (
                True  # Mark that a valid converged solution was found
            )
        else:
            patience_counter += 1  # Count iterations without improvement
        # print(patience_counter, best_converged_found)
        # Stop early if we have a converged best solution and no improvement in patience steps
        if patience_counter >= patience and best_converged_found:
            print(
                f"Final result taken from iteration {best_iteration} after {patience} iterations without improvement."
            )
            break

        # Early stopping criteria (only tracking objective stability now)
        if isinstance(early_stop, int):
            if early_stop > 0 and len(objective_history) > early_stop:
                max_change = np.max(
                    np.abs(np.diff(objective_history[-1 * early_stop :]))
                )
            if (
                early_stop > 0
                and len(objective_history) > early_stop
                and max_change < tol
                and converge
            ):
                print(
                    f"Final result taken from iteration {best_iteration}: "
                    f"Max objective change = {max_change:.6e}."
                )
                break

        # Update variables for next iteration
        A = A_new
        X = X_new
        Y = Y_new

    return X_best, history, objective_history, penalty_history, A_history
