import numpy as np
import matplotlib.pyplot as plt
from implementations import *
from .proj1_helpers import *

y_train, data_train, ids = load_csv_data("../data/train.csv")
y_test, data_test, ids = load_csv_data("../data/test.csv")


def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)


def compute_loss(y, tx, w):
    """Calculate the loss.
    You can calculate the loss using mse or mae.
    """
    e = y - tx.dot(w)
    return calculate_mse(e)


def expand_features(data):
    return np.append(np.ones((data.shape[0], 1)), data, axis=1)


def compute_accuracy(y, y_prediction):
    correct = 0.0
    for i in range (len(y)):
        if(y[i] == y_prediction[i]):
            correct += 1
    accuracy = correct/len(y)
    return accuracy


def plot_train_test(train_errors, test_errors, max_k_fold):
    """
    train_errors, test_errors and lambas should be list (of the same size) the respective train error and test error for a given lambda,
    * lambda[0] = 1
    * train_errors[0] = RMSE of a ridge regression on the train set
    * test_errors[0] = RMSE of the parameter found by ridge regression applied on the test set
    
    degree is just used for the title of the plot.
    """
    plt.plot(max_k_fold, train_errors, color='b', marker='*', label="Train error")
    plt.plot(max_k_fold, test_errors, color='r', marker='*', label="Test error")
    plt.xlabel("k_fold")
    plt.ylabel("MSE")
    plt.title("Regression")
    leg = plt.legend(loc=1, shadow=True)
    leg.draw_frame(False)
    plt.savefig("regression")


def cross_validation(y, x, k_fold, seed=1):
    """
    split the dataset in k equal subparts. Then train the model on k-1 parts and test it on the last remaining part.
    this process is repeatead k times with a different test part each time
    """
    assert(k_fold > 1)

    # set seed
    np.random.seed(seed)
    np.random.shuffle(x)
    # resetting the seed allows for an identical shuffling between y and x
    np.random.seed(seed)
    np.random.shuffle(y)
    
    loss_train = []
    loss_test = []
    
    total_length = len(y)
    split_length = int(total_length / k_fold)
    
    for i in range(k_fold):
        start_index = i * split_length
        if(i == k_fold - 1):
            end_index = total_length
        else: #this case is to ensure that all elements are present, in case of unbalanced split
            end_index = (i + 1) * split_length
        
        tx_train = x[np.r_[0 : start_index, end_index: total_length]]
        y_train = y[np.r_[0 : start_index, end_index: total_length]]

        tx_test = x[start_index : end_index]
        y_test = y[start_index : end_index]

        # linear regression
        weights = least_squares(y_train, tx_train)
        
        loss_train.append(compute_loss(y_train, tx_train, weights))
        loss_test.append(compute_loss(y_test, tx_test, weights))
        
    return np.mean(loss_train), np.mean(loss_test)


def run():  
    tx_train = expand_features(data_train)
    tx_test = expand_features(data_test)

    max_k_fold = 10
    losses_train = []
    losses_test = []
    for i in range(2, max_k_fold + 1):
        loss_train, loss_test = (cross_validation(y_train, tx_train, 10))
        losses_train.append(loss_train)
        losses_test.append(loss_test)
    plot_train_test(losses_train, losses_test, np.arange(2, max_k_fold + 1))

    weights_regression = least_squares(y_train, tx_train)  
    y_prediction = predict_labels(weights_regression, tx_train)

    print("Training set accuracy = "+str(compute_accuracy(y_train, y_prediction)))
    print("Loss = "+str(compute_loss(y_train, tx_train, weights_regression)))

    y_prediction_test = predict_labels(weights_regression, tx_test)
    create_csv_submission(ids, y_prediction_test, "submission.csv")


run()
