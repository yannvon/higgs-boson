import numpy as np

# ----- Helper functions linear regression -----------------------------------------------------------
def calculate_mse(e):
    """Calculate the mse for vector e."""
    #TODO FIXME did we just copy from the answer or did one of us come up with that ?
    return 1/2*np.mean(e**2)

def compute_loss_mse(y, tx, w):
    """Calculate the loss using mean squared error loss function"""
    e = y - tx @ w
    return calculate_mse(e)

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    N = len(y)
    e = y - tx @ w
    gradient = -(1.0 / N) * (tx.T @ e)
    return gradient, e

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    tx = np.empty([len(x), degree + 1])
    for i in range(len(x)):
        for j in range(degree + 1):
            tx[i, j] = x[i] ** (j+1)    
    return tx

# ----- Helper functions for logistic regression ----------------------------------------------------
def sigmoid(t):
    """apply sigmoid function on t."""
    return 1.0 / (1 + np.exp(-t))

def calculate_loss_logistic_regression(y, tx, w):
    """compute the cost by negative log likelihood."""
    return sum([np.log(1 + np.exp(tx[n].T @ w)) - y[n] * tx[n].T @ w for n in range(tx.shape[0])])

def calculate_gradient_logistic_regression(y, tx, w):
    """compute the gradient of loss."""
    return tx.T @ (sigmoid(tx @ w) - y)

# ----- Helper functions for penalized logistic regression -------------------------------------------
def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient, and hessian."""
    loss = calculate_loss_logistic_regression(y, tx, w) + lambda_ / 2 * sum(w @ w.T)
    gradient = calculate_gradient_logistic_regression(y, tx, w) + lambda_ * w # FIXME not sure with this one
    return loss, gradient

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
        print("loss", loss)
        weights = weights - gamma * gradient
    
    # return the last w and loss
    return weights, loss


# Linear regression using stochastic gradient descent
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent with batch size of 1"""
    np.random.seed(1)
    weights = initial_w
    min_weights = weights
    min_loss = calculate_mse(y - tx.dot(weights))
    for n_iter in range(max_iters):
        #Select a random element of y and tx
        r = np.random.randint(0, len(y))
        y_elem = np.array([y[r]])
        tx_elem = np.array([tx[r]])
        #Compute its stochastic gradient
        gradient, err = compute_gradient(y_elem, tx_elem, weights)
        #Update w
        weights = weights - gamma * gradient
        #Compute loss using mean squared error
        loss = calculate_mse(y - tx.dot(weights))
        if(loss < min_loss):
            min_loss = loss
            min_weights = weights
        #print("loss", loss)

    return min_weights, min_loss

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
    losses = [] # FIXME don't keep losses once tested
    threshold = 1e-8

    # build tx
    tx = np.c_[np.ones((y.shape[0], 1)), x]
    #w = np.zeros((tx.shape[1], 1))
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
    visualization(y, x, mean_x, std_x, w, "classification_by_logistic_regression_gradient_descent")
    print("loss={l}".format(l=calculate_loss(y, tx, w)))
    return w

# Regularized logistic regression using gradient descent or SGD
# FIXME taken straight from demo ex05, adapt convergence criterion
# FIXME print last loss
def reg_logistic_regression(y, tx, lambda_ , initial_w, max_iters, gamma):
    losses = [] # FIXME don't keep losses once tested
    threshold = 1e-8

    # build tx
    tx = np.c_[np.ones((y.shape[0], 1)), x]
    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, gradient = penalized_logistic_regression(y, tx, w)
        w = w - gamma * gradient
        
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    # visualization
    visualization(y, x, mean_x, std_x, w, "classification_by_logistic_regression_penalized_gradient_descent")
    print("loss={l}".format(l=calculate_loss(y, tx, w)))
    return w

# ----- Additional section: Newton method ---------------------------------------------------------------------------
# FIXME most functions taken straight from ex05, I still need to adapt them
# FIXME this Newton does not have the regularization term, easy to add !
def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""
    # compute S matrix
    N = tx.shape[0]
    S = np.zeros((N,N))
    for n in range(N):
        sig = sigmoid(tx[n].T @ w)
        S[n,n] = sig * (1 - sig)
    
    H = tx.T @ S @ tx
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
    return loss, w

# FIXME taken straight from ex05
# FIXME adapt names of subroutines called
def logistic_regression_newton_method_demo(y, x):
    # init parameters
    max_iter = 100
    threshold = 1e-8
    lambda_ = 0.1
    losses = []

    # build tx
    tx = np.c_[np.ones((y.shape[0], 1)), x]
    w = np.zeros((tx.shape[1], 1))

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_newton_method(y, tx, w)
        # log info
        if iter % 1 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    # visualization
    print("loss={l}".format(l=calculate_loss(y, tx, w)))
    return w


