#!/usr/bin/python
from __future__ import division
import numpy as np
import util
import operator
import math

# Returns the similarity between user i and user j
def compute_similarity(user_i, user_j):
    lambda4 = 80 # adjustable parameter
    age_difference = 10 # give a default age difference if unknown

    # If we are comparing the book against itself, pass
    if user_i == user_j:
        raise Exception('Cannot compare item against itself')

    if ages[user_i] != 0 and ages[user_j] != 0:
        age_difference = abs(ages[user_i] - ages[user_j])

    # Get set of items U that were rated both by user i and user j
    s1 = users[user_i].keys()
    s2 = users[user_j].keys()
    U = list(set(s1).intersection(set(s2)))
    n_ij = len(U) # Number of books rated by both user i and user j
    # Compute adjusted cosine similarity
    a = b = c = 0
    # for each book in U
    for isbn in U:
        # Compute average of the book's ratings
        R_mean = global_mean

        # Get user i's rating of the book
        R_ui = users[user_i][isbn]
        # Get user j's rating of the book
        R_uj = users[user_j][isbn]
        i_est = estimate_baseline(user_i, isbn) - R_mean
        j_est = estimate_baseline(user_j, isbn) - R_mean
        a += (i_est * j_est)
        b += pow(i_est, 2)
        c += pow(j_est, 2)
    try:
        sim = a / (math.sqrt(b) * math.sqrt(c))
        if (sim > 0):
            return n_ij / (lambda4 + 2*age_difference + n_ij) * sim # shrink correlation coefficient
    except:
        pass # Else no co-rated books
    

def estimate_baseline(user, isbn):
    return global_mean + user_baselines[user] + item_baselines[isbn]

# returns a sorted list of tuples with up to the k most similar users
def get_sim_users(sim_users, k):
    # sort the dictionary by decreasing similarity
    return sorted(sim_users.iteritems(), key=operator.itemgetter(1), reverse=True)[0:k]

pred_filename  = 'pred-user-based.csv'
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

# Dictionary of user ages
ages = {}

# Turn the list of users into a dictionary.
# Store data for each user to keep track of the per-user average.
# {user1: {isbn1: 4, isbn2: 5, ...}, user2: {...}, ...}
users = {}
for user in user_list:
    users[user['user']] = {}
    ages[user['user']] = user['age']

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

# For each book i in the testing set:
for query in test_queries:
    lambda5 = 1
    k = 100
    a = b = 0

    sim_users = {}
    user_i = query['user']
    isbn_i = query['isbn']
    # If the user is not in the training set, return baseline estimate of book
    if len(users[user_i].values()) == 0:
        query['rating'] = global_mean + item_baselines[isbn_i]
        continue
    # If the book is not in the training set, return baseline estimate of user
    elif len(items[isbn_i]) == 0:
        query['rating'] = global_mean + user_baselines[user_i]
        continue
    else: 
        baseline_rating = estimate_baseline(user_i, isbn_i)
        query['rating'] = baseline_rating

    for user_j in items[isbn_i]:
        sim = compute_similarity(user_i, user_j)
        if sim:
            sim_users[user_j] = sim

    neighborhood = get_sim_users(sim_users, k)
    for user_j, sim in neighborhood:
        r_uj = users[user_j][isbn_i]
        b_uj = estimate_baseline(user_j, isbn_i)
        a += sim * (r_uj - b_uj)
        b += abs(sim)

    # Take weighted average of the target user's ratings for the k most similar items            
    if len(neighborhood) > 10:
        # print len(neighborhood)
        query['rating'] = baseline_rating + lambda5 * a / b
        # print a / b
    # adjust if rating is over 5 or below 1
    if query['rating'] > 5:
        query['rating'] = 5
    if query['rating'] < 1:
        query['rating'] = 1

# Write the prediction file.
util.write_predictions(test_queries, pred_filename)