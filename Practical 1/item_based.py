#!/usr/bin/python
from __future__ import division
import numpy as np
import util
import math

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

# Returns the similarity between book i and book j
def compute_similarity(isbn_i, isbn_j):
    lambda4 = 100 # adjustable parameter
    # If we are comparing the book against itself, pass
    if isbn_i == isbn_j:
        raise Exception('Cannot compare item against itself')

    # Get set of users U that rated both book i and book j
    s1 = items[isbn_i].keys()
    s2 = items[isbn_j].keys()
    U = list(set(s1).intersection(set(s2)))
    n_ij = len(U) # Number of users that rated both book i and book j
    # Compute adjusted cosine similarity
    a = b = c = 0
    # for each user u in U
    for u in U:
        # Compute average of the u-th user's ratings
        R_mean = sum(users[u].values())/len(users[u].values())
        # R_mean_i = sum(items[isbn_i].values())/len(items[isbn_i].values())
        # R_mean_j = sum(items[isbn_j].values())/len(items[isbn_j].values())

        # Get user u's rating of book i
        R_ui = users[u][isbn_i]
        # Get user u's rating of book j
        R_uj = users[u][isbn_j]
        a += ((R_ui - R_mean) * (R_uj - R_mean))
        b += pow(R_ui - R_mean, 2)
        c += pow(R_uj - R_mean, 2)
    try:
        sim = a / (math.sqrt(b) * math.sqrt(c))
    except:
        return None
    if (sim > .25 and sim < 1):
        # print n_ij / (lambda4 + n_ij) * sim
        return n_ij / (lambda4 + n_ij) * sim # shrink correlation coefficient
    # Else no co-rated cases: return None

def estimate_baseline(user, isbn):
    x = global_mean + user_baselines[user] + item_baselines[isbn]
    return global_mean + user_baselines[user] + item_baselines[isbn]

# For each book i in the testing set:
for query in test_queries:
    sim_dict = {}
    user_i = query['user']
    isbn_i = query['isbn']
    # If the user is not in the training set, return baseline estimate of book
    if len(users[user_i].values()) == 0:
        query['rating'] = global_mean + item_baselines[isbn_i]
        pass
    # If the book is not in the training set, return baseline estimate of user
    elif len(items[isbn_i]) == 0:
        query['rating'] = global_mean + user_baselines[user_i]
        pass
    else: 
        baseline_rating = estimate_baseline(user_i, isbn_i)
        query['rating'] = baseline_rating

    # Look into the set of books the target user has rated and compute how similar
    # they are to the target book i
    try: 
        a = b = 0
        for isbn_j in users[user_i]:
            sim = compute_similarity(isbn_i, isbn_j)
            if sim:
                r_uj = users[user_i][isbn_j]
                b_uj = estimate_baseline(user_i, isbn_j)
                a += sim * (r_uj - b_uj)
                b += abs(sim)
        # Compute prediction by taking a weighted average of the target user's ratings
        # for the most similar items            
        query['rating'] = baseline_rating + a / b
        if query['rating'] > 5:
            print 'uh oh'
            query['rating'] = 5
        if query['rating'] < 1:
            query['rating'] = 1
        # print a / b
    # There are no similar items. Assign baseline rating.
    except:
        pass

# Write the prediction file.
util.write_predictions(test_queries, pred_filename)