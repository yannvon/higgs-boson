import numpy as np


# ----- Helper functions linear regression -----------------------------------------------------------
def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)


def compute_loss_mse(y, tx, w):
    """Calculate the loss using mean squared error loss function"""
    e = y - tx.dot(w)
    return calculate_mse(e)


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    N = len(y)
    e = y - tx @ w
    gradient = - (1.0 / N) * (tx.T @ e)
    return gradient, e


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    tx = np.empty([len(x), degree + 1])
    for i in range(len(x)):
        for j in range(degree + 1):
            tx[i, j] = x[i] ** (j+1)    
    return tx


# ----- Implement ML methods -------------------------------------------------------------------------
# Linear regression using gradient descent
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    weights = initial_w
    for n_iter in range(max_iters):
        # Compute gradient and loss
        gradient, e = compute_gradient(y, tx, weights)
        loss = calculate_mse(e)
        # Update w by gradient
        weights = weights- gamma * gradient
    
    # return the last w and loss
    return weights, loss


# Linear regression using stochastic gradient descent
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent with batch size of 1"""
    np.random.seed(1)
    weights = initial_w

    for n_iter in range(max_iters):
        # Select a random element of y and tx
        r = np.randomint(0, len(y))
        y_elem = np.array([y[r]])
        tx_elem = np.array([tx[r]])
        # Compute its stochastic gradient
        gradient, err = compute_gradient(y_elem, tx_elem, weights)
        # Update w
        weights = weights - gamma * gradient
        # Compute loss using mean squared error
        loss = calculate_mse(err)

    return weights, loss


# Least squares regression using normal equations
def least_squares(y, tx):
    """calculate the least squares solution."""
    # Returns mse, and optimal weights
    weights = np.linalg.inv((tx.T @ tx)) @ tx.T @ y
    return weights, compute_loss_mse(y, tx, weights)


# Ridge regression using normal equations
def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    l = 2 * tx.shape[0] * lambda_
    weights = np.linalg.inv(tx.T @ tx + l * np.identity(tx.shape[1])) @ tx.T @ y
    return weights, compute_loss_mse(y, tx, weights)


# Logistic regression using gradient descent or SGD
# FIXME taken straight from demo ex05, adapt convergence criterion
# FIXME print last loss
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    losses = []    # FIXME don't keep losses once tested
    threshold = 1e-8   # FIXME remove threshold

    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss = calculate_loss_logistic_regression(y, tx, w)
        gradient = calculate_gradient_logistic_regression(y, tx, w)
        w = w - gamma * gradient
        
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    # visualization
    loss = calculate_loss_logistic_regression(y, tx, w)
    print("loss={l}".format(l=loss))
    return w, loss


# Regularized logistic regression using gradient descent or SGD
# FIXME taken straight from demo ex05, adapt convergence criterion
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    losses = []    # FIXME don't keep losses once tested
    threshold = 1e-8   # FIXME remove threshold

    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        # FIXME cross check
        loss = calculate_loss_reg_logistic_regression(y, tx, w, lambda_)
        gradient = calculate_gradient_reg_logistic_regression(y, tx, w, lambda_)
        w = w - gamma * gradient
        
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    # visualization
    loss = calculate_loss_reg_logistic_regression(y, tx, w, lambda_)
    print("loss={l}".format(l=loss))
    return w, loss


# ----- Helper functions for logistic regression ----------------------------------------------------
def sigmoid(t):
    """apply sigmoid function on t."""
    return 1.0 / (1 + np.exp(-t))


def calculate_loss_logistic_regression(y, tx, w):
    """compute the cost by negative log likelihood."""
    # FIXME inspired by solutions since mine was bad.
    # return np.sum([np.log(1 + np.exp(tx[n].T @ w)) - y[n] * tx[n].T @ w for n in range(tx.shape[0])])
    return np.squeeze(-(y.T @ np.log(sigmoid(tx @ w)) + (1 - y).T @ np.log(1 - sigmoid(tx @ w))))


def calculate_gradient_logistic_regression(y, tx, w):
    """compute the gradient of loss."""
    return tx.T @ (sigmoid(tx @ w) - y)


# ----- Helper functions for penalized logistic regression -------------------------------------------
def calculate_loss_reg_logistic_regression(y, tx, w, lambda_):
    """compute the cost by negative log likelihood."""
    # FIXME lambda defined as in class !
    return calculate_loss_logistic_regression(y, tx, w) + lambda_ / 2 * np.squeeze(w.T @ w)


def calculate_gradient_reg_logistic_regression(y, tx, w, lambda_):
    """compute the gradient of loss."""
    # FIXME lambda defined as in class !
    return calculate_gradient_logistic_regression(y, tx, w) + lambda_ * w


# ----- Additional section: Newton method ---------------------------------------------------------------------------
# FIXME this Newton does not have the regularization term, easy to add !
def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""
    # compute S matrix
    N = tx.shape[0]
    S = np.zeros((N,N))
    for n in range(N):
        sig = sigmoid(tx[n].T @ w)
        S[n, n] = sig * (1 - sig)
    H = tx.T @ S @ tx
    # FIXME check for faster solution
    return H


def logistic_regression_newton(y, tx, w):
    """return the loss, gradient, and hessian."""
    loss = calculate_loss_logistic_regression(y, tx, w)
    gradient = calculate_gradient_logistic_regression(y, tx, w)
    hessian = calculate_hessian(y, tx, w)
    return loss, gradient, hessian


def learning_by_newton_method(y, tx, w):
    """
    Do one step on Newton's method.
    return the loss and updated w.
    """
    loss, gradient, hessian = logistic_regression_newton(y, tx, w)
    # We are not given gamma, so we assume that we should move to the position of the minimum
    w = w - np.linalg.inv(hessian) @ gradient
    return w, loss


# FIXME taken straight from ex05
# FIXME adapt names of subroutines called
def logistic_regression_newton_method_demo(y, tx, lambda_, initial_w, max_iters, gamma):
    # init parameters
    threshold = 1e-8
    losses = []

    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        w, loss = learning_by_newton_method(y, tx, w)

        # log info
        if iter % 1 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    # visualization
    loss = calculate_loss_logistic_regression(y, tx, w)
    print("loss={l}".format(l=loss))
    return w, loss

