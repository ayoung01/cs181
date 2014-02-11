# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 16:03:06 2014

Type: model

baseline model

rating = u + bi + bu

shrinkage towards 0 is applied through lambda2 and lambda3


@author: vincentli2010
"""

from __future__ import division
import util
import numpy as np

user_list      = util.user_list
book_list      = util.book_list

def baseline(train, test, mode='validation', param=0):
    
    # Compute the mean rating.
    train_mean = float(sum(map(lambda x: x['rating'], train)))/len(train)
    #train_mean = util.global_mean
    #train_mean  = 4.210194973215356    
    
    # Turn the list of users into a dictionary.
    # Store data for each user to keep track of the per-user average.
    # {user1: {isbn1: 4, isbn2: 5, ...}, user2: {...}, ...}
    users = {}
    for user in user_list:
        users[user['user']] = {}   
        
    # {isbn1: {user1: 4, user2: 5, ...}, isbn2: {...}, ...}    
    items = {}
    for item in book_list:
        items[item['isbn']] = {}
        
    for item in train:
        users[item['user']][item['isbn']] = item['rating']
        items[item['isbn']][item['user']] = item['rating']
    
    #lambda2_candidates = np.arange(50)
    #15, 3.5, 0.7749015226558873
    #14.2, 2.8 0.7747012569584306
    #16, 2.9 calibrated to sparse subset 0.7873733707932150
    lambda2 = 16 #item
    lambda3 = 2.9 #user
    # {isbn1: -.03, isbn2: .13, ...}    
    
    item_baselines = {}
    for isbn, ratings in items.iteritems():
        a = 0.0
        # Sum all differences between rating of item and global mean
        for rating in ratings.values():
            a += (rating - train_mean)
        item_baselines[isbn] = a / (lambda2 + len(ratings))
    
    # {user1: .215, user2: -.16, ...}
    user_baselines = {}
    for user, ratings in users.iteritems():
        a = 0.0
        # Sum r_ui - global_mean - baseline_i
        # ratings = {isbn1: 4, isbn2: 5, ...}
        for isbn, rating in ratings.iteritems():
            a += (rating - train_mean - item_baselines[isbn])
        user_baselines[user] = a / (lambda3 + len(ratings))     
        
    def predict(data, mode):
        y_hat = np.zeros((len(data), 1))
        if mode == 'rss': y = np.zeros((len(data), 1))           
        for i, entry in enumerate(data):
            isbn = entry['isbn']; user = entry['user'];
            bi = item_baselines[isbn]; bu = user_baselines[user];
            value = float(train_mean + bi + bu);  
            if mode == 'rss': y[i] = entry['rating']
            if value < 0:
                y_hat[i] = 0
            elif value > 5:
                y_hat[i] = 5
            else:
                y_hat[i] = value  
        if mode == 'rss':                     
            return float(np.linalg.norm(y-y_hat)**2)
        else:
            return y_hat
    
    if mode == 'prediction':
        return predict(test, 'y_hat'), predict(train, 'rss')
    if mode == 'validation':
        return predict(train, 'rss'), predict(test, 'rss')
                    
"""                    
    def validation_rmse(prediction, validation):
        res = []  
        for i in range(len(prediction)):
            res.append(float(prediction[i]['rating'] - validation[i]['rating']))
        res = np.array(res)
        return float(np.sqrt(np.dot(res.T,res)/len(res)))
    
    
    train_predict = []
    for entry in train:
        train_predict.append(entry.copy())
        
    validation_predict = []
    for entry in validation_data:
        validation_predict.append(entry.copy())
        
    #predict(train_predict)
    #train_rmse = validation_rmse(train_predict, training_data)
    
    predict(validation_predict)
    rmse = validation_rmse(validation_predict, validation_data)
        
    #print(train_rmse)
    print(rmse)
           
    #base_line_RMSE = 0.7813774640908429
          
    
    predict(test_queries)
    pred_filename  = 'pred-sgb-bias.csv'
    util.write_predictions(test_queries, pred_filename)
    """