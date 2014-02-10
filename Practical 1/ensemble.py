# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 16:20:34 2014

Linearly combine predictions from the following models:
baseline.py, baseline_l1.py, baseline_l2.py, sgd_bias.py, baseline_freq.py


@author: vincentli2010
"""
import numpy as np

import util
import baseline as bs
import baseline_l1 as bsl1
import baseline_l2 as bsl2
import sgd_bias as sgd
import baseline_freq as bsfreq

train_filename = 'data/ratings-train.csv'
test_filename  = 'data/ratings-test.csv'
user_filename  = 'data/users.csv'
book_filename  = 'data/books.csv'

train_valid    = util.load_train(train_filename)
test           = util.load_test(test_filename)
user_list      = util.load_users(user_filename)
book_list      = util.load_books(book_filename)


########################

num_model = 5

def run_models(train, valid):
    return (bs.baseline(train, valid), 
            bsl1.baseline_l1(train, valid), 
            bsl2.baseline_l2(train, valid), 
            sgd.sgd_bias(train, valid),
            bsfreq.baseline_freq(train,valid,'ensemble'))
 
pred_filename  = 'predictions/ensemble_v2_baselien_freq.csv'
#######################
 
n = len(train_valid)
p = num_model

x = np.zeros((n, p))
y = np.zeros((n, 1))
for i, entry in enumerate(train_valid):
        y[i] = float(entry['rating'])
        
def build_matrix(m, v, fold, model):
    span = len(v)
    m[fold*span:(fold+1)*span, model] = np.array(v)
        
       
# build linear ensemble
for k in range(5):        
    print 'Training ensemble fold: %d' % k
    span = n // 5
    valid = train_valid[k * span : (k + 1) * span]
    if k == 0: 
        train = train_valid[span:]
    elif k == 4:
        train = train_valid[:(len(train_valid)-span)]
    else:
        train = train_valid[: k * span]
        train.extend(train_valid[(k + 1) * span :])        
        
     
    model_results = run_models(train, valid)    
    for i, result in enumerate(model_results):
        build_matrix(x, result, fold = k, model = i)

#import pickle
#pickle.dump(x, open('data/x.p', 'wb')) 
b = np.linalg.solve(np.dot(x.T,x), np.dot(x.T,y))   
#b = np.linalg.lstsq(x, y)   

# make and combine predictions with ensemble
print ('Building final model')
predictions = np.zeros((len(test), num_model))
train = train_valid

model_results = run_models(train, test)  
  
for i, result in enumerate(model_results):
    build_matrix(predictions, result, fold = 0, model = i)
    
#pickle.dump(predictions, open('data/predictions.p', 'wb')) 

final_pred = np.dot(predictions, b)

for i, yi in enumerate(final_pred):
    if yi < 0:
        final_pred[i] = 0
    if yi > 5:
        final_pred[i] = 5

for i, entry in enumerate(test):
    entry['rating'] = float(final_pred[i])


util.write_predictions(test, pred_filename) 