import numpy as np

from scripts.proj1_helpers import *
from scripts.implementations import *
import matplotlib.pyplot as plt

y_train, data_train, ids = load_csv_data("../data/train.csv")
y_test, data_test, ids = load_csv_data("../data/test.csv")


def compute_accuracy(y, y_prediction):
    correct = 0.0
    for i in range(len(y)):
        if y[i] == y_prediction[i]:
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


# replace -999 value by the mean of the corresponding feature without taking the aberrant values into account
def remove_aberrant(data):
    for i in range(data.shape[1]):
        data[:, i][data[:, i] == -999.0] = np.mean(data[:, i][data[:, i] != -999.0])

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
        if i == k_fold - 1:
            end_index = total_length
        else: #this case is to ensure that all elements are present, in case of unbalanced split
            end_index = (i + 1) * split_length
        
        tx_train = x[np.r_[0 : start_index, end_index: total_length]]
        y_train = y[np.r_[0 : start_index, end_index: total_length]]

        tx_test = x[start_index : end_index]
        y_test = y[start_index : end_index]

        # logistic regression
        #weights, loss = logistic_regression(y, tx_train, initial_w, max_iters, lambda_)
        
        #loss_train.append(loss)
        #loss_test.append(calculate_loss_logistic_regression(y_test, tx_test, weights))
        
    return np.mean(loss_train), np.mean(loss_test)


def run():
    print("-- start test --")
    tx_train = np.c_[np.ones((y_train.shape[0], 1)), data_train]
    tx_test = np.c_[np.ones((y_test.shape[0], 1)), data_test]

    tx_train_cut = tx_train[np.r_[0, 10000]]
    y_train_cut = y_train[np.r_[0, 10000]]

    k_fold = 4
    # loss_train, loss_test = cross_validation(y_train, tx_train, 10)

    # logistic regression
    initial_w = np.zeros((tx_train.shape[1], 1))
    lambda_ = 0.1
    gamma = 0.01
    max_iters = 100000

    weights, loss = reg_logistic_regression(y_train_cut, tx_train_cut, lambda_, initial_w, max_iters, gamma)
    print("weights = " + str(weights))
    print("Loss = "+str(loss))

    y_prediction = predict_labels(weights, tx_train)
    print("y_prediction = " + str(y_prediction))
    print("Training set accuracy = "+str(compute_accuracy(y_train, y_prediction)))

    # Create submission
    y_prediction_test = predict_labels(weights, tx_test)
    create_csv_submission(ids, y_prediction_test, "submission.csv")


run()
