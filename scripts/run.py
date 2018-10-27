import numpy as np
from proj1_helpers import *
from implementations import *
from helpers import *
import matplotlib.pyplot as plt

def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)

def compute_loss(y, tx, w):
    """Calculate the loss.
    You can calculate the loss using mse or mae.
    """
    e = y - tx.dot(w)
    return calculate_mse(e)

def least_squares(y, tx):
    """calculate the least squares solution."""
    return np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))

def expand_features(x):
    return np.append(np.ones((x.shape[0], 1)), x, axis=1)

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

# replace -999 value by the mean of the corresponding feature without taking the aberrant values into account

def splitData(y, x):
    y_t0 = y[x[:, 22] == 0]
    y_t1 = y[x[:, 22] == 1]
    y_t23 = y[(x[:, 22] > 1)]
    x_t0 = x[x[:, 22] == 0][:, [0, 1, 2, 3, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21]]
    x_t1 = x[x[:, 22] == 1][:, [0, 1, 2, 3, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 29]]
    x_t23 = x[(x[:, 22] > 1)]
    return y_t0, y_t1, y_t23, x_t0, x_t1, x_t23

def standardize(x):
    """Standardize the original data set."""
    print(x)
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x

def cross_validation(y, x, k_fold, model, *args):
    """
    split the dataset in k equal subparts. Then train the model on k-1 parts and test it on the last remaining part.
    this process is repeatead k times with a different test part each time. The mean loss on the train and test sets are returned.
    the evaluation score is accuracy
    """
    assert(k_fold > 1)

    # set seed
    seed = 1
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
        weights = model(y_train, tx_train, *args)
        
        #TODO evaluate accuracy instead
        loss_train.append(compute_loss(y_train, tx_train, weights))
        loss_test.append(compute_loss(y_test, tx_test, weights))
        
    return np.mean(loss_train), np.mean(loss_test)

def run(): 
    # Load data
    y_train, tx_train, ids_train = load_csv_data("train.csv")
    y_test, tx_test, ids_test = load_csv_data("test.csv") 
   
    # Add a column of ones
    #tx_train = expand_features(tx_train)
    #tx_test = expand_features(tx_test)

    # Replace the -999 value in the first column by the mean
    tx_test[:, 0][tx_test[:, 0] == -999.0] = np.mean(tx_test[:, 0][tx_test[:, 0] != -999.0])
    tx_train[:, 0][tx_train[:, 0] == -999.0] = np.mean(tx_train[:, 0][tx_train[:, 0] != -999.0])

    # standardize the data (de-mean and divide by standard deviation)
    #tx_train = standardize(tx_train)
    #tx_test = standardize(tx_test)

    #k_fold of 10 has been empiricaly shown to yield a good bias-variance trade-off
    k_fold = 10
    #print(cross_validation(y_train, tx_train, k_fold, least_squares))
    
    weights = least_squares(y_train, tx_train)
    y_pred = predict_labels(weights, tx_train)
    print("Accuracy of basic least squares: \n", compute_accuracy(y_train, y_pred))

    y_t0, y_t1, y_t23, x_t0, x_t1, x_t23 = splitData(y_train, tx_train)
    w_0 = least_squares(y_t0, x_t0)
    w_1 = least_squares(y_t1, x_t1)
    w_2 = least_squares(y_t23, x_t23)
    y_pred = predict_labels_datasets(w_0, w_1, w_2, tx_train)
    print("Accuracy of least squares with 3 models, depending on the value of jet_num feature: \n", compute_accuracy(y_train, y_pred))
    
    #weights, loss = least_squares_SGD(y_train, tx_train, np.zeros(tx_train.shape[1]), 1000, 0.01)

    #y_prediction_test = predict_labels(weights, tx_test)
    #create_csv_submission(ids, y_prediction_test, "submission.csv")

run()
