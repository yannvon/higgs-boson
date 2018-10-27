import numpy as np

# ----- Helper functions -----------------------------------------------------------------------------
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
        #Select a random element of y and tx
        r = np.randomint(0, len(y))
        y_elem = np.array([y[r]])
        tx_elem = np.array([tx[r]])
        #Compute its stochastic gradient
        gradient, err = compute_gradient(y_elem, tx_elem, weights)
        #Update w
        weights = weights - gamma * gradient
        #Compute loss using mean squared error
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

#Logistic function on x
def logistic_function(x):
	return 1.0/(1-np.exp(-x))

# Logistic regression using gradient descent or SGD
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    pass
    #TODO

# Regularized logistic regression using gradient descent or SGD
def reg_logistic_regression(y, tx, lambda_ , initial_w, max_iters, gamma):
    pass
    #TODO


# ----- Additional methods ---------------------------------------------------------------------------




