# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 16:20:34 2014

Type: driver
Analyze dense and sparse subsets of the data using different models

@author: vincentli2010
"""

import util

user_list      = util.user_list
book_list      = util.book_list


def prediction(train_valid, test, pred_filename):
    
    import data_processing as dp
    dphelper = dp.data_processing()
    dense_train, sparse_train = dphelper.split(train_valid)
    dense_test, sparse_test = dphelper.split(test)
        
    #######
    import sgd_bias as sgd
    y_hat_dense, train_rmse_dense = sgd.sgd_bias(dense_train, dense_test, 'prediction')
    
    import baseline as bs
    y_hat_sparse, train_rmse_sparse = bs.baseline(sparse_train, sparse_test, 'prediction')
    
    #######
    print 'dense subset train rmse: %.16f' % train_rmse_dense
    print 'sparse subset train rmse: %.16f' % train_rmse_sparse
    test = dphelper.merge(test, y_hat_dense, y_hat_sparse)
    util.write_predictions(test, pred_filename) 

def validation(train, valid, mode='validation', param=0):
    
    import data_processing as dp
    dphelper = dp.data_processing()
    dense_train, sparse_train = dphelper.split(train)
    dense_valid, sparse_valid = dphelper.split(valid)
    
    
    import sgd_bias as sgd
    train_rss_dense, valid_rss_dense = sgd.sgd_bias(dense_train, dense_valid, 'validation')
    
    import baseline as bs
    train_rss_sparse, valid_rss_sparse = bs.baseline(sparse_train, sparse_valid, 'validation')
  
    return train_rss_dense + train_rss_sparse, valid_rss_dense + valid_rss_sparse
    
    
    