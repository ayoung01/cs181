# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import util
import operator
import math
# import matplotlib.pyplot as plt

user_list      = util.user_list
book_list      = util.book_list

pred_filename  = 'pred-amazon-baseline.csv'
train_filename = 'data/ratings-train.csv'
test_filename  = 'data/ratings-test.csv'
user_filename  = 'data/users.csv'
book_filename  = 'data/books.csv'

train  = util.load_train(train_filename)
test_queries   = util.load_test(test_filename)
user_list      = util.load_users(user_filename)
book_list      = util.load_books(book_filename)

# Compute the mean rating (4.070495)
train_mean = float(sum(map(lambda x: x['rating'], train)))/len(train)

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

# 929264 books indexed by isbn
# amazon_data = util.load_amazon('/Users/ayoung/Desktop/Books.txt')
import pickle
# pickle.dump(amazon_data, open('amazon_parsed.p', 'wb'))
amazon_data = pickle.load(open('amazon_parsed.p', 'rb'))
print 'amazon data loaded!'
refined_data = {}
for book, rating in amazon_data.iteritems():
    if book in items.keys():
        refined_data[book] = rating
pickle.dump(refined_data, open('amazon_refined.p', 'wb'))

amazon_baselines = {}
for isbn, ratings in amazon_data.iteritems():
    a = 0.0
    # Sum all differences between rating of item and global mean
    a += sum(ratings) - len(ratings) * amazon_mean
    amazon_baselines[isbn] = a / (lambda2 + len(ratings))

amazon_mean = 4.210194973215356

#lambda2_candidates = np.arange(50)
#15, 3.5, 0.7749015226558873
#14.2, 2.8 0.7747012569584306
#16, 2.9 calibrated to sparse subset 0.7873733707932150
lambda_t = 1.0
lambda2 = 16 #item
lambda3 = 2.9 #user
# {isbn1: -.03, isbn2: .13, ...}    


ratings_by_year = {} # {year1: [1,2,3], year2: [...]}
years = []; year_averages = [];
for book in book_list:
    year = book['year']
    isbn = book['isbn']
    if year > 2014 or year == 0:
        continue
    if not year in ratings_by_year:
        ratings_by_year[year] = []
    for value in items[isbn].values():
        ratings_by_year[year].append(value)

for year, ratings in ratings_by_year.iteritems():
    years.append(year)
    if (len(ratings) == 0):
        year_averages.append(0)
    else:
        year_averages.append(sum(ratings)/len(ratings))

year_averages = {} # {year1: 4.2, year2: 4.3, ...}
for year, ratings in ratings_by_year.iteritems():
    year_averages[year] = sum(ratings)/len(ratings)

year_baselines = {}
# for each book in training set
for year, ratings in ratings_by_year.iteritems():
    a = 0.0
    for rating in ratings.values():
        a += (rating - train_mean) 
    year_baselines[year] = a / (lambda_t + len(ratings))   



item_baselines = {}
for isbn, ratings in items.iteritems():
    a = 0.0
    # Sum all differences between rating of item and global mean
    year_effect = year_baselines[item_years[isbn]]
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



#     # If the user is not in the training set, return baseline estimate of book
#     if len(users[user_i].values()) == 0:
#         query['rating'] = train_mean + item_baselines[isbn_i]
#         continue
#     # If the book is not in the training set, return baseline estimate of user
#     elif len(items[isbn_i]) == 0:
#         query['rating'] = train_mean + user_baselines[user_i]
#         continue
#     else:
#         baseline_rating = estimate_baseline(user_i, isbn_i)
#         query['rating'] = baseline_rating
#     # adjust if rating is over 5 or below 1
#     if query['rating'] > 5:
#         query['rating'] = 5
#     if query['rating'] < 1:
#         query['rating'] = 1

# util.write_predictions(test_queries, pred_filename)