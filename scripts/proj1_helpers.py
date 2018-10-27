# -*- coding: utf-8 -*-
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


#Generate the predictions given the weigth of the data set with num jet 0, 1  or {2,3}
def predict_labels_datasets(weight0, weight1, weight23, data):
    ids = np.arange(data.shape[0])
    #For num jet 0
    indexFeatures0 = [0, 1, 2, 3, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    data0 = data[data[:, 22] == 0][:, indexFeatures0]
    ids0 = ids[data[:, 22] == 0]
    y_pred0 = np.dot(data0, weight0)
    #For num jet 1
    indexFeatures1 = [0, 1, 2, 3, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 29]
    data1 = data[data[:, 22] == 1][:, indexFeatures1]
    ids1 = ids[data[:, 22] == 1]
    y_pred1 = np.dot(data1, weight1)
    #For num jet 2,3
    data23 = data[data[:, 22] > 1]
    ids23 = ids[data[:, 22] > 1]
    y_pred23 = np.dot(data23, weight23)
    #Combining everything
    y_pred = np.concatenate((np.concatenate((y_pred0, y_pred1), axis=None),y_pred23),axis=None)
    ids = np.concatenate((np.concatenate((ids0, ids1), axis=None),ids23),axis=None)
    y = np.transpose(np.array([ids,y_pred]))
    y = y[y[:,0].argsort()][:,1]
    y[np.where(y <= 0)] = -1
    y[np.where(y > 0)] = 1
    return y

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

# Taken from labs
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
