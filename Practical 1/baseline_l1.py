# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 00:02:32 2014

y  = u + bi + bu with l1 penalty on bi and bu

@author: vincentli2010
"""
import util
import numpy as np
import scipy.sparse as ss
from sklearn import linear_model

user_filename  = 'data/users.csv'
book_filename  = 'data/books.csv'
user_list      = util.load_users(user_filename)
book_list      = util.load_books(book_filename)

def baseline_l1(train, test): 
    i_id = []; id_i = {}
    i = 0
    for entry in user_list:
        user = entry['user']
        i_id.append(user)
        id_i[user] = i
        i += 1
    for entry in book_list:
        isbn = entry['isbn']
        i_id.append(isbn)
        id_i[isbn] = i
        i += 1
        
    train_mean = float(sum(map(lambda x: x['rating'], train)))/len(train)
    
    def initialize(data, mode):
        # mapping index to user name and book isbn   
        n = len(data); p = len(user_list) + len(book_list)
        x = ss.lil_matrix((n,p), dtype = np.uint16)
        y = np.zeros((n,1), dtype = np.float)
        
        for row, entry in enumerate(data):
            user = entry['user']; isbn = entry['isbn'] 
            user_i = id_i[user]; isbn_i = id_i[isbn]        
            x[row, user_i] = np.uint16(1)
            x[row, isbn_i] = np.uint16(1)
            
            if mode == 'train':
                rating = entry['rating']
                y[row] = np.float(rating)
            
        if mode == 'train':
            y -= train_mean    
            x = ss.csc_matrix(x)
            #xtx = x.transpose().dot(x)
            #xty = x.transpose().dot(y)        
            return x, y
        else:
            return x
        
    # data preparation
    x_train, y_train = initialize(train, mode='train')
    x_test = initialize(test, mode='test')
    
    #alpha = 1.05e-5, test_rmse = 0.781118948510655
    alpha= 1.05e-5
    
    # solve l2
    clf = linear_model.Lasso(alpha=alpha, fit_intercept=False, normalize=False)
    clf.fit(x_train, y_train)
    b = clf.coef_; b = b.reshape((len(b), 1))
     
    # make prediction on test set
    y_predict = x_test.dot(b)
    for i, yi in enumerate(y_predict):
        if yi < 0:
            y_predict[i] = 0
        if yi > 5:
            y_predict[i] = 5
            
    for entry in test:
        isbn = entry['isbn']; user = entry['user'];
        bi = b[id_i[isbn]]; bu = b[id_i[user]];
        value = float(train_mean + bi + bu);
        if value < 0:
            entry['rating'] = 0
        elif value > 5:
            entry['rating'] = 5
        else:
            entry['rating'] = value
    
    return test    






























"""
ALPHA = np.arange(1e-5, 1e-4, 0.5e-6)
for alpha in ALPHA:
    clf = linear_model.Lasso(alpha=alpha, fit_intercept=False, normalize=False)
    clf.fit(train_x, train_y)
    b = clf.coef_; b = b.reshape((len(b), 1))
    print(np.sum(b))

    y_predict = train_x.dot(b)
    res = y_predict - train_y
    train_rmse = float(np.sqrt(np.dot(res.T, res)/len(res)))
    
    y_predict = valid_x.dot(b)
    res = y_predict - valid_y
    valid_rmse = float(np.sqrt(np.dot(res.T, res)/len(res)))
    
    print (alpha, train_rmse, valid_rmse)



"""









