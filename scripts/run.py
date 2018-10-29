import numpy as np
from split import *
from implementations import *
from helpers import *
from cross_validation import *

def run(): 
    # Load data
    y_train, tx_train, ids_train = load_csv_data("train.csv")
    y_test, tx_test, ids_test = load_csv_data("test.csv") 
   
    # Set degree of polynomial expansion
    degree = 8

    # Obtain different data sets depending on jet num
    y_t0, y_t1, y_t23 = split_y(y_train, tx_train)
    x_t0, x_t1, x_t23 = transform_x(tx_train, degree)

    # Train 3 models on each of the data sets
    w_0, loss = least_squares(y_t0, x_t0)
    w_1, loss = least_squares(y_t1, x_t1)
    w_2, loss = least_squares(y_t23, x_t23)

    # Predict labels for the train set and compute accuracy on train set
    y_pred = predict_labels_datasets(w_0, w_1, w_2, tx_train, transform_x, degree)
    print("Accuracy of least squares on train set: ", compute_accuracy(y_train, y_pred))

    # Compute mean accuracy and maximum deviation using 5-fold cross-validation (uncomment to use)
    #cross_result = cross_validation(y_train, tx_train, 5, degree, least_squares)
    #print("Least squares accuracy with 5-fold cross-validation = ", cross_result[0], "+-", cross_result[1])
    
    # Predict labels for the test set (Kaggle submission)
    y_pred = predict_labels_datasets(w_0, w_1, w_2, tx_test, transform_x, degree)

    # Create the actual submission
    create_csv_submission(ids_test, y_pred, "submission.csv")

run()
