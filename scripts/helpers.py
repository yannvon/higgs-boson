# -*- coding: utf-8 -*-
# ----- HELPER FUNCTIONS PROVIDED --------------------------------------------------------------------------------------
"""some helper functions for project 1."""
import csv
import numpy as np
from split import *

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
def compute_accuracy(y, y_prediction):
    ''' computes accuracy of a prediction '''
    correct = 0.0
    for i in range (len(y)):
        if(y[i] == y_prediction[i]):
            correct += 1
    accuracy = correct/len(y)
    return accuracy


def remove_aberrant(data):
    ''' removes missing values from dataset'''
    for i in range(data.shape[1]):
        data[:, i][data[:, i] == -999.0] = np.mean(data[:, i][data[:, i] != -999.0])


def build_poly(x, degree):
    """polynomial basis functions for input data x"""
    tx = x
    for i in range(2, degree + 1):
        power = np.apply_along_axis(lambda c: c**i, 1 , x)
        tx = np.concatenate((tx, power), axis=1)
    return tx

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
