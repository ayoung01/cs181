# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 16:20:34 2014

Linearly combine predictions from the following models:
baseline.py, baseline_l1.py, baseline_l2.py, sgd_bias.py


@author: vincentli2010
"""
import numpy as np

import util
import baseline as bs
import baseline_l1 as bsl1
import baseline_l2 as bsl2
import sgd_bias as sgd

train_filename = 'data/ratings-train.csv'
test_filename  = 'data/ratings-test.csv'
user_filename  = 'data/users.csv'
book_filename  = 'data/books.csv'

train_valid    = util.load_train(train_filename)
test           = util.load_test(test_filename)
user_list      = util.load_users(user_filename)
book_list      = util.load_books(book_filename)

num_model = 4
n = len(train_valid)
p = num_model

x = np.zeros((n, p))
y = np.zeros((n, 1))
for i, entry in enumerate(train_valid):
        y[i] = float(entry['rating'])
        
def build_matrix(m, v, fold, model):
    span = len(v)
    m[fold*span:(fold+1)*span, model] = np.array(v)
        
def run_models(train, valid):
    return (bs.baseline(train, valid), 
            bsl1.baseline_l1(train, valid), 
            bsl2.baseline_l2(train, valid), 
            sgd.sgd_bias(train, valid))
        
# build linear ensemble
for k in range(5):
    print 'Pass: %d' % k
    valid = train_valid[k * 40000 : (k + 1) * 40000]
    if k == 0: 
        train = train_valid[40000:]
    elif k == 4:
        train = train_valid[:160000]
    else:
        train = train_valid[: k * 40000]
        train.extend(train_valid[(k + 1) * 40000 :])
     
    x_bs, x_bsl1, x_bsl2, x_sgd = run_models(train, valid)
    
    build_matrix(x, x_bs, fold = k, model = 0)
    build_matrix(x, x_bsl1, fold = k, model = 1)
    build_matrix(x, x_bsl2, fold = k, model = 2)
    build_matrix(x, x_sgd, fold = k, model = 3)

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