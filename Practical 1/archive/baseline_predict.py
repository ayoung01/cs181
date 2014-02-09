# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 16:03:06 2014

@author: vincentli2010
"""

from __future__ import division
import util
import numpy as np
from matplotlib import pyplot as plt


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

#lambda2_candidates = np.arange(50)
lambda2 = 14.2 #item
lambda3 = 2.8 #user
# {isbn1: -.03, isbn2: .13, ...}    
item_baselines = {}
 # Adjustable parameter
for isbn, ratings in items.iteritems():
    a = 0
    # Sum all differences between rating of item and global mean
    for rating in ratings.values():
        a += (rating - global_mean)
    item_baselines[isbn] = a / (lambda2 + len(ratings))

# {user1: .215, user2: -.16, ...}
user_baselines = {}
for user, ratings in users.iteritems():
    a = 0
    # Sum r_ui - global_mean - baseline_i
    # ratings = {isbn1: 4, isbn2: 5, ...}
    for isbn, rating in ratings.iteritems():
        a += (rating - global_mean - item_baselines[isbn])
    user_baselines[user] = a / (lambda3 + len(ratings))

def predict(queries):
    for entry in queries:
        isbn = entry['isbn']; user = entry['user'];
        bi = item_baselines[isbn]; bu = user_baselines[user];     
        value = float(global_mean + bi + bu);
        if value < 0:
            entry['rating'] = 0
        elif value > 5:
            entry['rating'] = 5
        else:
            entry['rating'] = value

predict(test_queries)
# Write the prediction file.
pred_filename  = 'pred-baseline-14.2-2.8.csv'
util.write_predictions(test_queries, pred_filename)