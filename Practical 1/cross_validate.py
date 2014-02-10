# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 16:20:34 2014

cross_validation

@author: vincentli2010
"""
import numpy as np

import util
import baseline as bs
from matplotlib import pyplot as plt

train_filename = 'data/ratings-train.csv'
test_filename  = 'data/ratings-test.csv'
user_filename  = 'data/users.csv'
book_filename  = 'data/books.csv'

train_valid    = util.load_train(train_filename)
test           = util.load_test(test_filename)
user_list      = util.load_users(user_filename)
book_list      = util.load_books(book_filename)

num_folds = 1 # always 5-fold cross-validate, this decides how many folds to run
def run_model(train, valid, mode, param):
    return bs.baseline(train, valid, mode, param)
PARAM = []  

n = int(num_folds * (len(train_valid) // 5)) # e.g. length of num_folds * 40000
r = len(PARAM)
SCORE = np.zeros((r, 2)) #1st column train, 2nd column valid

# cross validation
for k in range(num_folds):
    print 'Pass: %d' % k
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
        train_rmse, valid_rmse = run_model(train, valid, mode = 'cv', param)
        SCORE[i, 0] = train_rmse; SCORE[i, 1] = valid_rmse

for i in range(r):
    print 'Param: %f\tTrain: %.16\tValidation: %.16' % (PARAM[i], SCORE[i, 0], SCORE[i, 1])
if r > 1:
    plt.plot(PARAM, SCORE)

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