# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 00:02:32 2014

y  = u + bi + bu with l2 penalty on bi and bu

@author: vincentli2010
"""
import util
import numpy as np
import scipy.sparse as ss
import scipy.sparse.linalg as ssl

user_filename  = 'data/users.csv'
book_filename  = 'data/books.csv'
user_list      = util.load_users(user_filename)
book_list      = util.load_books(book_filename)

def baseline_l2(train, test): 
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
    x_train, y_train = initialize(train, 'train')
    x_test = initialize(test, 'test')
    
    # lambda = 4.8, tuned by cross-validation
    lamb = 4.8 
    
    # solve l2
    sol = ssl.lsqr(x_train, y_train, damp = np.sqrt(lamb), iter_lim = 100000)
    b = np.array(sol[0]).reshape((len(sol[0]), 1))
    
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
    # cross-validation
    LAMB = np.arange(1, 10, 0.1)
    RSS = np.zeros((6, len(LAMB)))
    
    for k in range(5):
        valid = train_valid[k * 40000 : (k + 1) * 40000]
        if k == 0: 
            train = train_valid[40000:]
        elif k == 4:
            train = train_valid[:160000]
        else:
            train = train_valid[: k * 40000]
            train.extend(train_valid[(k + 1) * 40000 :])
        x_train, y_train = initialize(train)
        x_valid, y_valid = initialize(valid)
        
        for i, lamb in enumerate(LAMB):
            sol = ssl.lsqr(x_train, y_train, damp = np.sqrt(lamb), iter_lim = 100000)
            b = np.array(sol[0]).reshape((len(sol[0]), 1))
            y_predict = x_valid.dot(b)
            for i, yi in enumerate(y_predict):
                if yi < 0:
                    y_predict[i] = 0
                if yi > 5:
                    y_predict[i] = 5
            res = y_predict - y_valid
            rmse = float(np.dot(res.T, res))
            RSS[k, i] = rmse
    
    for i in range(len(LAMB)):
        RSS[5, i] = np.sqrt(np.sum(RSS[0:5,i]) / len(train_valid))
    
    index = np.where(RSS[5,:]==np.min(RSS[5,:]))
    print float(index[0])/10 + 1 #lamb = 4.8
    print np.min(RSS[5,:]) #rss4.8 = 0.774799805324
    plt.plot(LAMB, RSS[5,:])
    """
    
    
    #prediction
    """
    def predict(train, queries, lamb):
        x_train, y_train = initialize(train)
        sol = ssl.lsqr(x_train, y_train, damp = np.sqrt(lamb), iter_lim = 100000)
        b = np.array(sol[0]).reshape((len(sol[0]), 1))
        
        # hard-thresholding b
        delta = 0.02
        for i, bi in enumerate(b):
            if np.abs(bi) < delta:
                b[i] = 0    
        
        for entry in queries:
            isbn = entry['isbn']; user = entry['user'];
            bi = b[id_i[isbn]]; bu = b[id_i[user]];
            value = float(global_mean + bi + bu);
            if value < 0:
                entry['rating'] = 0
            elif value > 5:
                entry['rating'] = 5
            else:
                entry['rating'] = value
                
    predict(train_valid, test_queries, 4.8)
    # Write the prediction file.
    pred_filename  = 'baseline_l2_4.8_v2.csv' #0.774799805324
    util.write_predictions(test_queries, pred_filename)
    """