# -*- coding: utf-8 -*-
# ----- HELPER FUNCTIONS PROVIDED --------------------------------------------------------------------------------------
"""some helper functions for project 1."""
import csv
import numpy as np


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    return y_pred

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})


# ----- ADDITIONAL HELPER FUNCTIONS ------------------------------------------------------------------------------------
#Generate the predictions given the weigth of the data set with num jet 0, 1  or {2,3}
def predict_labels_datasets(weight0, weight1, weight23, data, transform_x):
    ids = np.arange(data.shape[0])

    tx_0, tx_1, tx_23 = transform_x(data)

    #For num jet 0
    ids0 = ids[data[:, 22] == 0]
    y_pred0 = np.dot(tx_0, weight0)
 
    #For num jet 1
    ids1 = ids[data[:, 22] == 1]
    y_pred1 = np.dot(tx_1, weight1)

    #For num jet 2,3
    ids23 = ids[data[:, 22] > 1]
    y_pred23 = np.dot(tx_23, weight23)

    #Combining everything
    y_pred = np.concatenate((np.concatenate((y_pred0, y_pred1), axis=None),y_pred23),axis=None)
    ids = np.concatenate((np.concatenate((ids0, ids1), axis=None),ids23),axis=None)
    y = np.transpose(np.array([ids,y_pred]))
    y = y[y[:,0].argsort()][:,1]
    y[np.where(y <= 0)] = -1
    y[np.where(y > 0)] = 1
    return y

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

def split_y(y, x):
    y_t0 = y[x[:, 22] == 0]
    y_t1 = y[x[:, 22] == 1]
    y_t23 = y[(x[:, 22] > 1)]
    return y_t0, y_t1, y_t23

def split_x(x):
    x_t0 = x[x[:, 22] == 0][:, [0, 1, 2, 3, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21]]
    x_t1 = x[x[:, 22] == 1][:, [0, 1, 2, 3, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 29]]
    x_t23 = x[(x[:, 22] > 1)]
    return x_t0, x_t1, x_t23



def compute_accuracy(y, y_prediction):
    correct = 0.0
    for i in range (len(y)):
        if(y[i] == y_prediction[i]):
            correct += 1
    accuracy = correct/len(y)
    return accuracy

def remove_aberrant(data):
    for i in range(data.shape[1]):
        data[:, i][data[:, i] == -999.0] = np.mean(data[:, i][data[:, i] != -999.0])


# ----- FUNCTIONS TAKEN FROM LAB ---------------------------------------------------------------------------------------
def sample_data(y, x, seed, size_samples):
    """sample from dataset."""
    np.random.seed(seed)
    num_observations = y.shape[0]
    random_permuted_indices = np.random.permutation(num_observations)
    y = y[random_permuted_indices]
    x = x[random_permuted_indices]
    return y[:size_samples], x[:size_samples]


def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x

def build_poly(x, degree):
    """polynomial basis functions for input data x"""
    tx = x
    for i in range(2, degree + 1):
        power = np.apply_along_axis(lambda c: c**i, 1 , x)
        tx = np.concatenate((tx, power), axis=1)
    return tx
