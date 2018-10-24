import numpy as np
from proj1_helpers import *

y_train, data_train, ids = load_csv_data("train.csv")
y_test, data_test, ids = load_csv_data("test.csv")

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

def expand_features(data):
    return np.append(np.ones((data.shape[0], 1)), data, axis=1)

def compute_accuracy(y, y_prediction):
    correct = 0.0
    for i in range (len(y)):
        if(y[i] == y_prediction[i]):
            correct += 1
    accuracy = correct/len(y)
    return accuracy

def run():  
    tx_train = expand_features(data_train)
    tx_test = expand_features(data_test)

    weights_regression = least_squares(y_train, tx_train)  
    y_prediction = predict_labels(weights_regression, tx_train)
    

    print("Training set accuracy = "+str(compute_accuracy(y_train, y_prediction)))
    print("Loss = "+str(compute_loss(y_train, tx_train, weights_regression)))

    y_prediction_test = predict_labels(weights_regression, tx_test)
    create_csv_submission(ids, y_prediction_test, "submission.csv")

run()
