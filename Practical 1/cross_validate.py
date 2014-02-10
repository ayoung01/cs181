# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 16:20:34 2014

cross_validation

@author: vincentli2010
"""
import numpy as np
import util
from matplotlib import pyplot as plt

user_list      = util.user_list
book_list      = util.book_list
train_filename = 'data/ratings-train.csv'
train_valid    = util.load_train(train_filename)

# split data into dense and sparse subsets
import data_processing as dp
dphelper = dp.data_processing(train_valid)
dense_subset, sparse_subset = dphelper.split()

# work on the dense subset
#train_valid = dense_subset

# work on the sparse subset
train_valid = sparse_subset

######### Tuning Parameters #########

PARAM = [10]
#PARAM = np.arange(1, 50, 5) 

num_folds = 5 # always 5-fold cross-validate, this decides how many folds to run

import baseline as bs
def run_model(train, valid, mode, param):
    return bs.baseline(train, valid, mode, param)

#####################################
n = int(num_folds * (len(train_valid) // 5)) # e.g. length of num_folds * 40000
r = len(PARAM)
SCORE = np.zeros((r, 2)) #1st column train, 2nd column valid
# cross validation
for k in range(num_folds):
    print 'Fold: %d' % k
    span = n // num_folds
    valid = train_valid[k * span : (k + 1) * span]
    if k == 0: 
        train = train_valid[span:]
    elif k == 4:
        train = train_valid[:(len(train_valid)-span)]
    else:
        train = train_valid[: k * span]
        train.extend(train_valid[(k + 1) * span :])
    
    for i, param in enumerate(PARAM):
        # model_result 
        train_rss, valid_rss = run_model(train, valid, 'validation', param)
        SCORE[i, 0] += train_rss; SCORE[i, 1] += valid_rss

def calc_rmse(rss, mode):
    if mode == 'train':
        num_data_points = int(num_folds * 4 * (len(train_valid) // 5))
    elif mode == 'valid':
        num_data_points = n
    else:
        num_data_points = 0
    return float(np.sqrt(rss/num_data_points))

def display():
    RMSE = np.zeros((r, 2))
    for i in range(r):
        RMSE[i, 0] = calc_rmse(SCORE[i, 0], 'train')
        RMSE[i, 1] = calc_rmse(SCORE[i, 1], 'valid')
        print 'Param: %f Train: %.16f Valid: %.16f' % (
            PARAM[i], RMSE[i, 0], RMSE[i, 1])
    
    idx_hat = np.argmin(RMSE[:,1])
    print '******\nParam: %f Train: %.16f Valid: %.16f\n******' % (
            PARAM[idx_hat], RMSE[idx_hat, 0], RMSE[idx_hat, 1])
    
    if r > 1:
        trainplot, = plt.plot(PARAM, RMSE[:, 0])
        validplot, = plt.plot(PARAM, RMSE[:, 1])
        plt.title("RMSE Plot")    
        plt.legend([trainplot, validplot], ["Train", "Valid"])

display()

"""
x = np.zeros((n, r, 2)) # 2 layers, 1 for train predictions and 1 for valid predictions
y = np.zeros((n, 1))
for i, entry in enumerate(train_valid):
        y[i] = float(entry['rating'])

       
def build_matrix(m, v, fold_idx, param_idx):
    span = np.shape(v)[0] 
    # train predictions
    m[fold_idx*span:(fold_idx+1)*span, param_idx, 0] = np.array(v[:, 0])
    # valid predictions    
    m[fold_idx*span:(fold_idx+1)*span, param_idx, 1] = np.array(v[:, 1])
"""

"""
#import pickle
#pickle.dump(x, open('data/x.p', 'wb')) 
b = np.linalg.solve(np.dot(x.T,x), np.dot(x.T,y))   
#b = np.linalg.lstsq(x, y)   

# make and combine predictions with ensemble
predictions = np.zeros((len(test), num_model))
train = train_valid

pred_bs, pred_bsl1, pred_bsl2, pred_sgd = run_models(train, test)
build_matrix(predictions, pred_bs, fold = 0, model = 0)
build_matrix(predictions, pred_bsl1, fold = 0, model = 1)
build_matrix(predictions, pred_bsl2, fold = 0, model = 2)
build_matrix(predictions, pred_sgd, fold = 0, model = 3)
#pickle.dump(predictions, open('data/predictions.p', 'wb')) 

final_pred = np.dot(predictions, b)

for i, yi in enumerate(final_pred):
    if yi < 0:
        final_pred[i] = 0
    if yi > 5:
        final_pred[i] = 5

for i, entry in enumerate(test):
    entry['rating'] = float(final_pred[i])

pred_filename  = 'predictions/ensemble_v1_excludel1.csv'
util.write_predictions(test, pred_filename)
""" 