# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import util
import operator
import math
import matplotlib.pyplot as plt

def estimate_baseline(user, isbn):
    x = train_mean + user_baselines[user]
    if isbn in amazon_baselines:
        print 'yay'
        return x + amazon_baselines[isbn]
    else:
        print 'nay'
        return x + item_baselines[isbn]

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

# 929264 books indexed by isbn
amazon_data = util.load_amazon('/Users/ayoung/Desktop/Books.txt')
import pickle
pickle.dump(amazon_data, open('amazon_parsed.p', 'wb'))
# amazon_data = pickle.load(open('amazon_parsed.p', 'rb'))
print 'amazon data loaded!'

# # Compute amazon global mean:
# sum_ratings = num_ratings = 0
# for (score, helpfulness) in amazon_data.itervalues():
#     sum_ratings += sum(score)
#     num_ratings += len(score)
# amazon_mean = sum_ratings / num_ratings

amazon_mean = 4.210194973215356

#lambda2_candidates = np.arange(50)
#15, 3.5, 0.7749015226558873
#14.2, 2.8 0.7747012569584306
#16, 2.9 calibrated to sparse subset 0.7873733707932150
lambda2 = 16 #item
lambda3 = 2.9 #user
# {isbn1: -.03, isbn2: .13, ...}    

amazon_baselines = {}
for isbn, ratings in amazon_data.iteritems():
    a = 0.0
    # Sum all differences between rating of item and global mean
    a += sum(ratings) - len(ratings) * amazon_mean
    amazon_baselines[isbn] = a / (lambda2 + len(ratings))

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
    # Sum r_ui - train_mean - baseline_i
    # ratings = {isbn1: 4, isbn2: 5, ...}
    for isbn, rating in ratings.iteritems():
        a += (rating - train_mean - item_baselines[isbn])
    user_baselines[user] = a / (lambda3 + len(ratings))

# {age: [4, 4, 3, ...] ..., age2: [...]}
ratings_by_age = {}
ages = []; averages = [];
for user in user_list:
    user_age = user['age']
    user_id = user['user']
    if user_age == 0 or user_age > 100:
        continue
    if not user_age in ratings_by_age:
        ratings_by_age[user_age] = []
    for value in users[user_id].values():
        ratings_by_age[user_age].append(value)

for age, ratings in ratings_by_age.iteritems():
    ages.append(age)
    averages.append(sum(ratings)/len(ratings))

# {year1: [1,2,3], year2: [...]}
ratings_by_year = {}
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

amazon_num = []
train_num = []

for query in test_queries:
    user_i = query['user']
    isbn_i = query['isbn']
    # If book is in amazon dataset and book has two or fewer ratings:
    if isbn in amazon_baselines: #and len(items[isbn]) < 3:
        # Get number of ratings of book:
        num_train_ratings = len(items[isbn])
        # Get number of ratings book has on amazon:
        num_amazon_ratings = len(amazon_baselines[isbn])
        amazon_num.append(num_amazon_ratings)
        train_num.append(num_train_ratings)
# Plot differences
plt.plot(amazon_num, train_num)
plt.savefig('amazonvstrain.png')

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