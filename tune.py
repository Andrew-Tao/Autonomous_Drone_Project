import numpy as np
from numpy.linalg import lstsq
import random

from controller import controller
import simulate


def tune():
    """
    Compute an optimal set of PID parameters by trying multiple attempts
    to minimize the cost function and selecting the best result.

    Returns:
        np.array: The best set of PID parameters found.
    """
    attempts = 10
    min_cost = 1e10
    best_theta = None

    for _ in range(attempts):
        theta, costs = minimize()
        if costs[-1] < min_cost:
            min_cost = costs[-1]
            best_theta = theta

    return best_theta


def minimize():
    """
    Minimize the cost function using numerical gradient estimation and gradient descent.

    Returns:
        np.array: Optimized PID parameters.
        list: Costs at each iteration.
    """
    theta = np.random.rand(3)  # Initialize weights randomly
    alpha = 0.03  # Step size
    max_iterations = 500
    average_length = 3
    costs = []

    for iteration in range(1, max_iterations + 1):
        print(f"Iteration {iteration}...")

        # Compute the cost with averaging
        current_cost = mean_value(cost, theta, average_length)
        costs.append(current_cost)

        # Check for steady state
        if iteration > 55:  # num_costs + 5
            recent_costs = costs[-50:]  # Previous 50 costs
            X = np.vstack([np.ones(50), np.arange(1, 51)]).T
            b, residuals, rank, s = lstsq(X, np.array(recent_costs), rcond=None)
            slope_ci = 1.96 * np.std(recent_costs) / np.sqrt(len(recent_costs))

            if -slope_ci <= b[1] <= slope_ci:
                break

        # Adjust step size and averaging
        if iteration > 100:
            alpha = 0.001
            average_length = 8
        if iteration > 200:
            average_length = 15
            alpha = 0.0005

        # Compute gradient and update parameters
        grad = mean_value(gradient, theta, average_length)
        theta -= alpha * grad

    return theta, costs


def mean_value(func, input, n):
    """
    Compute the average of a function evaluated multiple times with randomness.

    Args:
        func (callable): The function to evaluate.
        input (np.array): Input to the function.
        n (int): Number of evaluations.

    Returns:
        float: The averaged value.
    """
    values = [func(input) for _ in range(n)]
    return sum(values) / n


def gradient(theta):
    """
    Estimate the gradient of the cost function numerically.

    Args:
        theta (np.array): Current PID parameters.

    Returns:
        np.array: Gradient vector.
    """
    delta = 0.001
    grad = np.zeros(len(theta))
    random.seed(42)  # Set a fixed seed for consistent disturbance

    for i in range(len(theta)):
        var = np.zeros(len(theta))
        var[i] = 1

        left_cost = cost(theta + delta * var)
        right_cost = cost(theta - delta * var)
        grad[i] = (left_cost - right_cost) / (2 * delta)

    return grad


def cost(theta):
    """
    Compute the cost function for a given set of PID parameters.

    Args:
        theta (np.array): PID parameters.

    Returns:
        float: Cost value.
    """
    control = controller('pid', theta[0], theta[1], theta[2])
    data = simulate(control, 0, 1, 0.05)
    errors = np.sqrt(np.sum(data["theta"] ** 2, axis=1))
    return np.sum(errors ** 2) * data["dt"]
