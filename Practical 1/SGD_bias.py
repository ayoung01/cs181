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

# initialize feature vectors
feature_dimension = 5
gamma = 0.1 # learning rate
lamb = 0.1 #regularization
initial = 0.01

initial_value = initial * np.ones((feature_dimension, 1))
for user in user_list:
    users[user['user']]['p'] = initial_value
for item in book_list:
    items[item['isbn']]['q'] = initial_value

# main gradient decsend algorithm - one epoch
def descend():
    for entry in training_data:
        rating = entry['rating']; isbn = entry['isbn']; user = entry['user'];
        bi = item_baselines[isbn]; bu = user_baselines[user];
        q = items[isbn]['q']; p = users[user]['p'];
        
        # model
        r_hat = global_mean + bi + bu + np.dot(q.T, p)
        
        e = rating - r_hat
            
        items[isbn]['q'] = q + gamma * (e * p - lamb * q)
        users[user]['p'] = p + gamma * (e * q - lamb * p)

def descend_gain(items_old, users_old, items, users):
    items_gain = 0; users_gain = 0; 
    for key in items_old:
        items_gain += np.linalg.norm(items_old[key]['q'] - items[key]['q'])
    for key in users_old:
        users_gain += np.linalg.norm(users_old[key]['p'] - users[key]['p'])
    return items_gain, users_gain

def print_feature(feature, name):    
    for pos in range(5):
        print feature[feature.keys()[pos]][name]
        print len(feature[feature.keys()[pos]])-1
    

for i in range(30):
    descend()
    
    
print_feature(users, 'p')     
print(descend_gain(items_old, users_old, items, users))       
        

# Make predictions for each test query.
for entry in test_queries:
    isbn = entry['isbn']; user = entry['user'];
    bi = item_baselines[isbn]; bu = user_baselines[user];
    q = items[isbn]['q']; p = users[user]['p'];       
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

# Write the prediction file.
util.write_predictions(test_queries, pred_filename)
