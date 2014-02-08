#!/usr/bin/python
import numpy as np
import pandas
import util
import math
from collections import namedtuple

pred_filename  = 'pred-user-mean.csv'
train_filename = 'ratings-train.csv'
test_filename  = 'ratings-test.csv'
user_filename  = 'users.csv'
book_filename  = 'books.csv'

training_data = pandas.DataFrame.from_csv(train_filename)
test_queries = pandas.DataFrame.from_csv(test_filename)
user_list = pandas.DataFrame.from_csv(user_filename)
book_list = pandas.DataFrame.from_csv(book_filename)

# Returns the similarity between book i and book j
def compute_similarity(isbn_i, isbn_j):
    # If we are comparing the book against itself, pass
    if isbn_i == isbn_j:
        raise Exception('Cannot compare item against itself')

    # Get set of users U that rated both book i and book j
    s1 = training_data.loc[(training_data.ISBN == isbn_i)].User
    s2 = training_data.loc[(training_data.ISBN == isbn_j)].User
    U = list(set(s1).intersection(set(s2)))

    # Compute adjusted cosine similarity
    if len(U) > 0:
        a = b = c = 0
        # for each user u in U
        for u in U:
            # Compute average of the u-th user's ratings
            R_mean = training_data.loc[(training_data.User == u)].Rating.mean()
            # Get user u's rating of book i
            R_ui = training_data.loc[(training_data.ISBN == isbn_i) &
                                        (training_data.User == u)].Rating.iget(0)
            # Get user u's rating of book j
            R_uj = training_data.loc[(training_data.ISBN == isbn_j) &
                                        (training_data.User == u)].Rating.iget(0)
            a += (R_ui - R_mean) * (R_uj - R_mean)
            b += pow(R_ui - R_mean, 2)
            c += pow(R_uj - R_mean, 2)
        sim = a / (math.sqrt(b) * math.sqrt(c))
        if (sim > .5):
            return sim
    # Else no co-rated cases: return None
# For each book i in the testing set:
for count, query in test_queries.iterrows():
    sim_dict = {}
    user_i = query[0] 
    isbn_i = query[1]
    user_mean = training_data.loc[(training_data.User == user_i)].Rating.mean()
    rating = user_mean

    # If the book is not in the training set, assign default
    if len(training_data.loc[(training_data.ISBN == isbn_i)].ISBN) == 0:
        print 'not in training set'
        print rating
        pass

    # Look into the set of books the target user has rated and compute how similar
    # they are to the target book i
    for isbn_j in training_data.loc[(training_data.User == user_i)].ISBN:
        sim = compute_similarity(isbn_i, isbn_j)
        if sim:
            sim_dict[isbn_j] = sim

    # Compute prediction by taking a weighted average of the target user's ratings
    # for the most similar items
    if sim_dict:
        a = b = 0
        for isbn, sim in sim_dict.items():
            a += sim * training_data.loc[(training_data.ISBN == isbn)].Rating.iget(0)
            b += sim
        rating = abs(a / b)
        print rating - user_mean
    # There are no similar items. Assign default rating.
    else:
        print 'no similarity'
        print rating
# # Write the prediction file.
# util.write_predictions(test_queries, pred_filename)
