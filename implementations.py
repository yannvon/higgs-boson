import numpy as np
import random as rd

# ----- Helper functions -----------------------------------------------------------------------------
def compute_gradient(y, tx, w):
    """Compute the gradient."""
    N = len(y)
    e = y - tx @ w
    gradient = - (1.0 / N) * (tx.T @ e)
    return gradient, e

# FIXME remove this as it is the same as above method.
def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    N = len(y)
    e = y - tx @ w
    gradient = - (1.0 / N) * (tx.T @ e)
    return gradient, e

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # FIXME use different for construct for better speed / style

    # this function should return the matrix formed
    # by applying the polynomial basis to the input data

    tx = np.empty([len(x), degree + 1])
    for i in range(len(x)):
        for j in range(degree + 1):
            tx[i, j] = x[i] ** (j+1)    
    return tx


# ----- Implement ML methods -------------------------------------------------------------------------

# Linear regression using gradient descent
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    w = initial_w
    for n_iter in range(max_iters):
        # Compute gradient and loss
        gradient, e = compute_gradient(y, tx, w)
        loss = 1/2 * np.mean(e ** 2)
        # Update w by gradient
        w = w - gamma * gradient
    
    # return the last w and loss
    return w, loss


# Linear regression using stochastic gradient descent
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent with batch size of 1"""
    w = initial_w
    for n_iter in range(max_iters):
        #Select a random element of y and tx
        r = rd.randint(0,len(y)-1)
        y_elem = np.array([y[r]])
        tx_elem = np.array([tx[r]])
        #Compute its stochastic gradient
        gradient,err = compute_stoch_gradient(y_elem,tx_elem,w)
        #Update w
        w = w - gamma * gradient
    
    #FIXME Use method computeloss??
    loss = 1/2*np.mean(err**2)
    return w, loss

# Least squares regression using normal equations
def least_squares(y, tx):
    """calculate the least squares solution."""
    # returns mse, and optimal weights
    # FIXME return loss too
    w_optimal = np.linalg.inv((tx.T @ tx)) @ tx.T @ y
    return w_optimal

# Ridge regression using normal equations
def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    # FIXME return loss too
    # FIXME why do I get inconsistent results in the jupyter demo?
    # FIXME test this function
    l = 2 * tx.shape[0] * lambda_
    w_ridge = np.linalg.inv(tx.T @ tx + l * np.identity(tx.shape[1])) @ tx.T @ y
    return w_ridge


# Logistic regression using gradient descent or SGD
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    pass
    #TODO

# Regularized logistic regression using gradient descent or SGD
def reg_logistic_regression(y, tx, lambda_ , initial_w, max_iters, gamma):
    pass
    #TODO


# ----- Additional methods ---------------------------------------------------------------------------




