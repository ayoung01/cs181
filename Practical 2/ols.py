# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 14:47:08 2014

@author: vincentli2010
"""

import numpy as np
#import util

"""
X_train.shape = (1147, 61)
8 number_of_screens
11 production_budget

0 christmas_release
2 independence_release
3 labor_release
4 memorial_release
13 summer_release

1 highgest_grossing_actors_present
9 oscar_winning_actors_present
10 oscar_winning_directors_present

5 num_highest_grossing_actors
6 num_oscar_winning_actors
7 num_oscar_winning_directors

14 review word count
15-36 genres
37-49 companies
50-54 ratings

12 running_time
55-60 sentiments 

observations:
1. companies sharply divide high and low profile revenues 
"""    
import pickle as pickle        
X_train, global_feat_dict, y_train, train_ids = pickle.load(open('features.p', 'rb'))         

## Segmentation
#mask = X_train[:, 8] < 100
#mask = np.array([all(x) for x in zip(X_train[:,8]>100, X_train[:,8]<1000)])
mask = X_train[:, 8] > 1000

## transformation
# log y
y_train = np.log(y_train)

# number_of_screens
X_train[:, 8] = X_train[:, 8] ** 2

# production_budget
X_train[:, 11] = X_train[:, 11] ** 0.3


# discard highgest_grossing_actors_present and num_highest_grossing_actors because missing
pass

# discard num_oscar_winning_directors beacuse same as oscar_winning_directors_present
pass

# discard genres columns  under 6 counts
pass

# discard companies columns under 1 counts

# discard ratings collumns under 2 counts


display_set = [12]
y_plot = y_train[mask, np.newaxis]
X_plot = X_train[mask,:][:,display_set]



from pandas.tools.plotting import scatter_matrix
from pandas import DataFrame
df = DataFrame(np.concatenate((y_plot, X_plot), axis=1))
scatter_matrix(df, alpha=0.2, figsize=(15, 15), diagonal='kde')


