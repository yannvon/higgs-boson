# -*- coding: utf-8 -*-
import numpy as np
from split import *
from helpers import *
# ----- This file contains all methods needed to perform cross validation on any model ----------------------------------

def cross_validation(y, x, k_fold, degree, model, *args):
    """
    split the dataset in k equal subparts. Then train the model on k-1 parts and test it on the last remaining part.
    this process is repeatead k times with a different test part each time. The mean accuracy on the test set and the
    absolute value of the maxium deviation are returned.
    the evaluation metric is accuracy
    """
    assert(k_fold > 1)

    # set seed
    seed = 1
    np.random.seed(seed)
    np.random.shuffle(x)
    # resetting the seed allows for an identical shuffling between y and x
    np.random.seed(seed)
    np.random.shuffle(y)
    
    accuracies = []

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
        #weights, loss = model(y_train, tx_train, *args)

        y_t0, y_t1, y_t23 = split_y(y_train, tx_train)
        x_t0, x_t1, x_t23 = transform_x(tx_train, degree)

        w_0, l = model(y_t0, x_t0)
        w_1, l = model(y_t1, x_t1)
        w_2, l = model(y_t23, x_t23)

        y_prediction = predict_labels_datasets(w_0, w_1, w_2, tx_test, transform_x, degree)
        accuracy = compute_accuracy(y_test, y_prediction)
        print(accuracy)
        accuracies.append(accuracy)
     
    maximum_deviation = 0
    mean_accuracy = np.mean(accuracies)
    for accuracy in accuracies:
        deviation = np.abs(accuracy - mean_accuracy)
        if(deviation > maximum_deviation):
            maximum_deviation = deviation
    return mean_accuracy, maximum_deviation

def transform_x(x, degree):
    """
    Cleans the data according to the cleaning used for the best submission
    Returns 3 data sets depending on jet num, with polynomial expansion
    """

    # Step 1: Replace the -999 values in the first column by the mean
    x[:, 0][x[:, 0] == -999.0] = np.mean(x[:, 0][x[:, 0] != -999.0])
    
    # Step 2: Split data set into 3 datasets depending on jet num
    x_0, x_1, x_23 = split_x(x)
    
    # Step 3: Expand features with polynomial basis
    x_0 = build_poly(x_0, degree)
    x_1 = build_poly(x_1, degree)
    x_23 = build_poly(x_23, degree)

    # Final step: return all transformed data
    return x_0, x_1, x_23

