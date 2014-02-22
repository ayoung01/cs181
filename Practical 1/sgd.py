# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 16:03:06 2014

Type: model

Matrix Factorization approach, solved with Stochastic Gradient Descent

Model: 

rating = global_avg + user_dev + book_dev + user_feature.T * book_feature

@author: vincentli2010
"""

from __future__ import division
import util
import numpy as np

user_list      = util.user_list
book_list      = util.book_list


def sgd_bias(train, test, mode='validation', param=0): 
    
    # Compute the mean rating.
    train_mean = float(sum(map(lambda x: x['rating'], train)))/len(train)
    
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
    
    # {isbn1: -.03, isbn2: .13, ...}    
    item_baselines = {}
    lambda2 = 14.2 # Adjustable parameter
    for isbn, ratings in items.iteritems():
        a = 0.0
        # Sum all differences between rating of item and global mean
        for rating in ratings.values():
            a += (rating - train_mean)
        item_baselines[isbn] = a / (lambda2 + len(ratings))
    
    # {user1: .215, user2: -.16, ...}
    user_baselines = {}
    lambda3 = 2.8
    for user, ratings in users.iteritems():
        a = 0.0
        # Sum r_ui - train_mean - baseline_i
        # ratings = {isbn1: 4, isbn2: 5, ...}
        for isbn, rating in ratings.iteritems():
            a += (rating - train_mean - item_baselines[isbn])
        user_baselines[user] = a / (lambda3 + len(ratings))
    
    
    def descend():
        # train the first feature
        j = 0; 
        #descend_gain = 999
        for iteration in range(num_iter):
            #print("i")
            for entry in train:
                rating = entry['rating']; isbn = entry['isbn']; user = entry['user'];
                #bi = item_baselines[isbn]; bu = user_baselines[user];
                q = float(items[isbn]['q'][j]); p = float(users[user]['p'][j]);
                
                #r_hat = train_mean + bi + bu + q * p
                r_hat = train_mean + q * p
                e = rating - r_hat
                           
                items[isbn]['q'][j] = q + gamma * (e * p - lamb * q)
                users[user]['p'][j] = p + gamma * (e * q - lamb * p)
                entry['e_old'] = e            
            
        #predict(train_predict, 1)
        #train_rmse = validation_rmse(train_predict, train)
        #print(train_rmse)
        #TRAIN_RMSE.append(train_rmse)
    
        #predict(validation_predict, 1)
        #rmse = validation_rmse(validation_predict, validation_data)
        #print(rmse)
        #RMSE.append(rmse)
    
               
        for j in np.arange(1,feature_dimension):    
            for iteration in range(num_iter):
                #print("i")
                for entry in train:
                    isbn = entry['isbn']; user = entry['user'];
                    q = items[isbn]['q'][j]; p = users[user]['p'][j];
                    
                    e = entry['e_old'] - p * q                 
                    items[isbn]['q'][j] = q + gamma * (e * p - lamb * q)
                    users[user]['p'][j] = p + gamma * (e * q - lamb * p)
                    entry['e_old'] = e
            
            #predict(train_predict, j+1)
            #train_rmse = validation_rmse(train_predict, train)
            #print(train_rmse)
            #TRAIN_RMSE.append(train_rmse)   
            
            #predict(validation_predict, j+1)
            #rmse = validation_rmse(validation_predict, validation_data)
            #print(rmse)
            #RMSE.append(rmse)        
    
    def calc_descend_gain(items_hist, users_hist, items, users, j):
        items_res = []; users_res = [];
        for key in items_hist:
            items_res.append(float(items_hist[key]['q'][j] - items[key]['q'][j]))
        for key in users_hist:
            users_res.append(float(users_hist[key]['p'][j] - users[key]['p'][j]))   
        return np.sum(np.abs(items_res))/len(items_res), np.sum(np.abs(users_res))/len(users_res)
    
    def print_feature(feature, name):    
        for pos in range(50):
            print feature[feature.keys()[pos]][name]
            print len(feature[feature.keys()[pos]])-1
            
    def fprint(iterable):
        for item in iterable:
            print '%.16f' % item
                      
    def validation_rmse(prediction, validation):
        res = []  
        for i in range(len(prediction)):
            res.append(float(prediction[i]['rating'] - validation[i]['rating']))
        res = np.array(res)
        return float(np.sqrt(np.dot(res.T,res)/len(res)))  
    
    def predict(data, n, mode):
        y_hat = np.zeros((len(data), 1))
        if mode == 'rss': y = np.zeros((len(data), 1))           
        for i, entry in enumerate(data):
            isbn = entry['isbn']; user = entry['user'];
            bi = item_baselines[isbn]; bu = user_baselines[user];
            q = np.array(items[isbn]['q'][:n]); p = np.array(users[user]['p'][:n]);   
            if mode == 'rss': y[i] = entry['rating']
            if len(items[isbn]) == 1:
                value = float(train_mean + bi + bu);
                if value < 0:
                    y_hat[i] = 0
                elif value > 5:
                    y_hat[i] = 5
                else:
                    y_hat[i] = value
            else:
                #value = float(train_mean + bi + bu + np.dot(q.T, p))
                value = float(train_mean + np.dot(q.T, p))
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
 
 
    
    # main gradient decsend algorithm - one epoch
    #RMSE = []; TRAIN_RMSE = [];
    #validation_predict = []
    #for entry in validation_data:
    #    validation_predict.append(entry.copy())
        
    #train_predict = []
    #for entry in train:
    #    train_predict.append(entry.copy())
     
    #tuned to lamb = 0.05 with RMSE = 0.7802252543829444
    #lamb = 4.01 with RMSE = 0.7747012079725805
     
    ####################### 
    # lamb = 1 with RMSE = 0.7730765178141169 
    lamb = param #regularization
    feature_dimension = 5
    gamma = 0.1 # learning rate
    initial = 0.1
    num_iter = 10 #epoch 70 to converge
    
    ########################
    
    initial_value = [initial] * feature_dimension
    for user in user_list:
        users[user['user']]['p'] = list(initial_value)
    for item in book_list:
        items[item['isbn']]['q'] = list(initial_value)

    descend()   
    if mode == 'prediction':
        return predict(test, feature_dimension, 'y_hat'), predict(train, feature_dimension, 'rss')
    if mode == 'validation':
        return predict(train, feature_dimension, 'rss'), predict(test,feature_dimension, 'rss')
    
    #plt.plot(range(num_iter),RMSE)
    
    #print_feature(users, 'p') 
    
    #base_line_RMSE = 0.7813774640908429
          
    """
    predict(test_queries)
    # Write the prediction file.
    util.write_predictions(test_queries, pred_filename)
    """