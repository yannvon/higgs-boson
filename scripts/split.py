# -*- coding: utf-8 -*-
import numpy as np

# ----- This file contains all methods used to split the higgs boson data set into 3 parts, depending on jet num -------

def split_y(y, x):
    ''' 
    This method splits the labels into 3 arrays.
    X must be included in order to perform the correct split
    '''
    y_t0 = y[x[:, 22] == 0]
    y_t1 = y[x[:, 22] == 1]
    y_t23 = y[(x[:, 22] > 1)]
    return y_t0, y_t1, y_t23

def split_x(x):
    '''This method splits the data points into 3 arrays.'''
    x_t0 = x[x[:, 22] == 0][:, [0, 1, 2, 3, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21]]
    x_t1 = x[x[:, 22] == 1][:, [0, 1, 2, 3, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 29]]
    x_t23 = x[(x[:, 22] > 1)]
    return x_t0, x_t1, x_t23


def predict_labels_datasets(weight0, weight1, weight23, data, transform_x):
    ''' Generate the predictions given the weigth of the data set with num jet 0, 1  or {2,3} '''
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


def predict_labels_datasets_logistic(weight0, weight1, weight23, data, transform_x):
    ''' Generate the predictions given the weigth of the data set with num jet 0, 1  or {2,3} 
    This method is specific to the logistic regression, because it maps a probability to a value
    '''
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
    y[np.where(y <= 0.5)] = -1
    y[np.where(y > 0.5)] = 1
    return y

