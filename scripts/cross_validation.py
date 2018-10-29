# -*- coding: utf-8 -*-
import numpy as np

# ----- This file contains all methods needed to perform cross validation on any model ----------------------------------

def cross_validation(y, x, k_fold, model, *args):
    """
    split the dataset in k equal subparts. Then train the model on k-1 parts and test it on the last remaining part.
    this process is repeatead k times with a different test part each time. The mean loss on the train and 
    test sets are returned.
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

