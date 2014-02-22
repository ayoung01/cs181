# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 13:32:32 2014

Type: helper

Split a dataset into a dense and a sparse subset
Merge predictions for these two subsets

@author: vincentli2010
"""

import util
threshold_per_user = 50
threshold_per_book = 2

class data_processing:
    def __init__(self):
        pass
    
    def split(self, data):    
        self.dense = []; self.sparse = []   
        for item in data:
            user = item['user']; isbn = item['isbn']
            if len(util.users[user]) > threshold_per_user and len(util.books[isbn]) > threshold_per_book:
                self.dense.append(item)
            else:
                self.sparse.append(item)
        
        print '%.3f percent of data is dense' % (float(len(self.dense))/len(data))
        
        return self.dense, self.sparse
        
    def merge(self, test, y_dense, y_sparse):
        self.dense_counter = 0; self.sparse_counter = 0;        
        for entry in test:
            user = entry['user']; isbn = entry['isbn']
            if len(util.users[user]) > threshold_per_user and len(util.books[isbn]) > threshold_per_book:
                entry['rating'] = float(y_dense[self.dense_counter])
                self.dense_counter += 1
            else:
                entry['rating'] = float(y_sparse[self.sparse_counter])
                self.sparse_counter += 1
        return test
  
    """
    ratings_per_user = []
    for user in users.values():
        ratings_per_user.append(len(user))
    
    counter = 0
    for num in ratings_per_user:
        if num > 30:
            counter+=1
    print float(counter)/len(ratings_per_user)
        
    ratings_per_book = []
    for book in books.values():
        ratings_per_book.append(len(book))    
    
    counter = 0
    for num in ratings_per_book:
        if num > 1:
            counter+=1
    print float(counter)/len(ratings_per_book)
    """
    











