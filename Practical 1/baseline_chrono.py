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

def baseline_chrono(train, test, mode='validation', param=0):
  
    # Compute the mean rating.
    train_mean = float(sum(map(lambda x: x['rating'], train)))/len(train)
    #train_mean = util.global_mean
    #train_mean  = 4.210194973215356  
    
    # Turn the list of users into a dictionary.
    # Store data for each user to keep track of the per-user average.
    users = {} # {user1: {isbn1: 4, isbn2: 5, ...}, user2: {...}, ...}
    for user in user_list:
        users[user['user']] = {}   
    
    items = {} # {isbn1: {user1: 4, user2: 5, ...}, isbn2: {...}, ...} 
    for item in book_list:
        items[item['isbn']] = {}
    
    for item in train:
        users[item['user']][item['isbn']] = item['rating']
        items[item['isbn']][item['user']] = item['rating']
    
    item_years = {} # {isbn1: 1994, isbn2: 1990, ...}
    for item in book_list:
        item_years[item['isbn']] = item['year']
    
####################    
    lambda_t = 33000
    lambda2 = 14 #item
    lambda3 = 2.8 #user
    # {isbn1: -.03, isbn2: .13, ...}    
####################    
    
    ratings_by_year = {} # {year1: [1,2,3], year2: [...]
    for book in book_list:
        year = book['year']
        isbn = book['isbn']
        if not year in ratings_by_year:
            ratings_by_year[year] = []
        for value in items[isbn].values():
            ratings_by_year[year].append(value)

    year_baselines = {}
    # for each book in training set
    for year, ratings in ratings_by_year.iteritems():
        if year > 2014 or year == 0:
            year_baselines[year] = 0
        a = 0.0 
        for rating in ratings:
            a += (rating - train_mean) 
        year_baselines[year] = a / (lambda_t + len(ratings))   
 
    item_baselines = {}
    for isbn, ratings in items.iteritems():
        a = 0.0
        year = item_years[isbn]
        # Sum all differences between rating of item and global mean
        year_effect = year_baselines[year]
        for rating in ratings.values():
            a += (rating - train_mean - year_effect)
        item_baselines[isbn] = a / (lambda2 + len(ratings))
    
    # {user1: .215, user2: -.16, ...}
    user_baselines = {}
    for user, ratings in users.iteritems():
        a = 0.0
        # Sum r_ui - train_mean - baseline_i
        # ratings = {isbn1: 4, isbn2: 5, ...}
        year_effect = year_baselines[item_years[isbn]]
        for isbn, rating in ratings.iteritems():
            a += (rating - train_mean - year_effect - item_baselines[isbn])
        user_baselines[user] = a / (lambda3 + len(ratings))
    
    def predict(data, mode):
        y_hat = np.zeros((len(data), 1))
        if mode == 'rss': y = np.zeros((len(data), 1))           
        for i, entry in enumerate(data):
            isbn = entry['isbn']; user = entry['user'];
            year = item_years[isbn]
            bi = item_baselines[isbn]; bu = user_baselines[user];
            bt = year_baselines[year]
            value = float(train_mean + bt + bi + bu);  
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
  year_averages = [];   
    for year, ratings in ratings_by_year.iteritems():
        years.append(year)
        if (len(ratings) == 0):
            year_averages.append(0)
        else:
            year_averages.append(sum(ratings)/len(ratings))
    
    year_averages = {} # {year1: 4.2, year2: 4.3, ...}
   for year, ratings in ratings_by_year.iteritems():
        year_averages[year] = sum(ratings)/len(ratings)  
"""    
