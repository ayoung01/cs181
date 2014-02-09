# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 16:03:06 2014

@author: vincentli2010
"""

from __future__ import division
import util
import numpy as np

pred_filename  = 'pred-sgb-bias.csv'
train_filename = 'ratings-train.csv'
test_filename  = 'ratings-test.csv'
user_filename  = 'users.csv'
book_filename  = 'books.csv'

training_data  = util.load_train(train_filename)
test_queries   = util.load_test(test_filename)
user_list      = util.load_users(user_filename)
book_list      = util.load_books(book_filename)

validation_data = training_data[0:40000]
training_data = training_data[40000:]

# Compute the mean rating.
num_train = len(training_data)
global_mean = float(sum(map(lambda x: x['rating'], training_data)))/num_train

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
    
for item in training_data:
    users[item['user']][item['isbn']] = item['rating']
    items[item['isbn']][item['user']] = item['rating']

# {isbn1: -.03, isbn2: .13, ...}    
item_baselines = {}
lambda2 = 25 # Adjustable parameter
for isbn, ratings in items.iteritems():
    a = 0
    # Sum all differences between rating of item and global mean
    for rating in ratings.values():
        a += (rating - global_mean)
    item_baselines[isbn] = a / (lambda2 + len(ratings))

# {user1: .215, user2: -.16, ...}
user_baselines = {}
lambda3 = 10
for user, ratings in users.iteritems():
    a = 0
    # Sum r_ui - global_mean - baseline_i
    # ratings = {isbn1: 4, isbn2: 5, ...}
    for isbn, rating in ratings.iteritems():
        a += (rating - global_mean - item_baselines[isbn])
    user_baselines[user] = a / (lambda3 + len(ratings))


def descend():
    # train the first feature
    j = 0; 
    #descend_gain = 999
    for iteration in range(num_iter):
        #print("i")
        for entry in training_data:
            rating = entry['rating']; isbn = entry['isbn']; user = entry['user'];
            bi = item_baselines[isbn]; bu = user_baselines[user];
            q = float(items[isbn]['q'][j]); p = float(users[user]['p'][j]);
            
            r_hat = global_mean + bi + bu + q * p
            e = rating - r_hat
                       
            items[isbn]['q'][j] = q + gamma * (e * p - lamb * q)
            users[user]['p'][j] = p + gamma * (e * q - lamb * p)
            entry['e_old'] = e            
        
    predict(train_predict, 1)
    train_rmse = validation_rmse(train_predict, training_data)
    #print(train_rmse)
    TRAIN_RMSE.append(train_rmse)

    predict(validation_predict, 1)
    rmse = validation_rmse(validation_predict, validation_data)
    #print(rmse)
    RMSE.append(rmse)

           
    for j in np.arange(1,feature_dimension):    
        for iteration in range(num_iter):
            #print("i")
            for entry in training_data:
                isbn = entry['isbn']; user = entry['user'];
                q = items[isbn]['q'][j]; p = users[user]['p'][j];
                
                e = entry['e_old'] - p * q                 
                items[isbn]['q'][j] = q + gamma * (e * p - lamb * q)
                users[user]['p'][j] = p + gamma * (e * q - lamb * p)
                entry['e_old'] = e
        
        predict(train_predict, j+1)
        train_rmse = validation_rmse(train_predict, training_data)
        #print(train_rmse)
        TRAIN_RMSE.append(train_rmse)   
        
        predict(validation_predict, j+1)
        rmse = validation_rmse(validation_predict, validation_data)
        #print(rmse)
        RMSE.append(rmse)        

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

def predict(queries, n):
    for entry in queries:
        isbn = entry['isbn']; user = entry['user'];
        bi = item_baselines[isbn]; bu = user_baselines[user];
        q = np.array(items[isbn]['q'][:n]); p = np.array(users[user]['p'][:n]);       
        if len(items[isbn]) == 1:
            value = float(global_mean + bi + bu);
            if value < 0:
                entry['rating'] = 0
            elif value > 5:
                entry['rating'] = 5
            else:
                entry['rating'] = value
        else:
            value = float(global_mean + bi + bu + np.dot(q.T, p))
            if value < 0:
                entry['rating'] = 0
            elif value > 5:
                entry['rating'] = 5
            else:
                entry['rating'] = value  
                
def validation_rmse(prediction, validation):
    res = []  
    for i in range(len(prediction)):
        res.append(float(prediction[i]['rating'] - validation[i]['rating']))
    res = np.array(res)
    return float(np.sqrt(np.dot(res.T,res)/len(res)))


# main gradient decsend algorithm - one epoch
RMSE = []; TRAIN_RMSE = [];
validation_predict = []
for entry in validation_data:
    validation_predict.append(entry.copy())
    
train_predict = []
for entry in training_data:
    train_predict.append(entry.copy())
 

for lamb in [0.05]: 
#tuned to lamb = 0.05 with RMSE = 0.7802252543829444
    # initialize feature vectors
    TRAIN_RMSE.extend([lamb, lamb])
    RMSE.extend([lamb, lamb])
    feature_dimension = 4
    gamma = 0.1 # learning rate
    #lamb = 0.1 #regularization
    initial = 0.1
    num_iter = 5 #epoch
    
    initial_value = [initial] * feature_dimension
    for user in user_list:
        users[user['user']]['p'] = list(initial_value)
    for item in book_list:
        items[item['isbn']]['q'] = list(initial_value)

    descend()   
    
    print(TRAIN_RMSE)
    print(RMSE)        


#plt.plot(range(num_iter),RMSE)

#print_feature(users, 'p') 

#base_line_RMSE = 0.7813774640908429
      
"""
predict(test_queries)
# Write the prediction file.
util.write_predictions(test_queries, pred_filename)
"""