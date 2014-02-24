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

14 review word count 

55-60 sentiments 0.28247*** 58 -> 0.16029 # number of negative [0] sentences in review

61 np_release_dates
12 running_time

5 num_highest_grossing_actors
6 num_oscar_winning_actors
7 num_oscar_winning_directors

0 christmas_release
2 independence_release
3 labor_release
4 memorial_release
13 summer_release

1 highgest_grossing_actors_present
9 oscar_winning_actors_present
10 oscar_winning_directors_present

15-36 genres 
37-49 companies 
50-54 ratings 
observations:
1. companies sharply divide high and low profile revenues 
"""    
import pickle as pickle        
X_train_1, global_feat_dict, y_train, train_ids = pickle.load(open('features.p', 'rb'))         
X_train_2 = np.load(open('feat.npy', 'rb'))   
X_train = np.concatenate((X_train_1, X_train_2), axis=1)     

## Imputation
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values= -1, strategy='mean', axis=0)
X_train = imp.fit_transform(X_train)

## transformation
# log y
y_train = np.log(y_train)

# production_budget trasnform by taking power
POWER = [-0.3, 0.1, 0.3, 0.5, 1.5]
for power in POWER: 
    X_train = np.concatenate((X_train, X_train[:, 11][:, np.newaxis] ** power), axis=1)
X_train = np.concatenate((X_train, np.log(X_train[:, 11][:, np.newaxis])), axis=1)

# number_of_screens
POWER = [-2, -1, 0.1, 0.5, 1.5, 2]
for power in POWER: 
    X_train = np.concatenate((X_train, X_train[:, 8][:, np.newaxis] ** power), axis=1)
X_train = np.concatenate((X_train, np.log(X_train[:, 8][:, np.newaxis])), axis=1)

#14 review word count    
POWER = [-2, -1, 0.1, 0.5, 2, 3] 
for power in POWER: 
    X_train = np.concatenate((X_train, X_train[:, 14][:, np.newaxis] ** power), axis=1)
X_train = np.concatenate((X_train, np.log(X_train[:, 14][:, np.newaxis])), axis=1)

#sentiments: avg, numpos, numneg
POWER = [-8, -7, -5, -4, -3, -2, -1, -0.1 ,0.1, 0.5, 2, 3, 5, 7, 8, 9]
for i in [55, 57, 58]:
    for power in POWER: 
        X_train = np.concatenate((X_train, (X_train[:, i][:, np.newaxis] + 1) ** power), axis=1)
    X_train = np.concatenate((X_train, np.log(X_train[:, i][:, np.newaxis] + 1)), axis=1)

# running time
POWER = [-2, -1, 0.1, 0.5, 2] 
for power in POWER: 
    X_train = np.concatenate((X_train, X_train[:, 12][:, np.newaxis] ** power), axis=1)
X_train = np.concatenate((X_train, np.log(X_train[:, 12][:, np.newaxis])), axis=1)


# np_release_dates
POWER = [-2, -1, 0.1, 0.5, 2, 3] 
for power in POWER: 
    X_train = np.concatenate((X_train, (X_train[:, 61][:, np.newaxis] + 1) ** power), axis=1)
X_train = np.concatenate((X_train, np.log(X_train[:, 61][:, np.newaxis] + 1)), axis=1)


"""
import scipy.special as ss
for i in [56, 59, 60]:
    X_train = np.concatenate((X_train, ss.logit(X_train[:, i][:, np.newaxis]+0.01)), axis=1)
"""



"""
# 5 num_highest_grossing_actors
# 6 num_oscar_winning_actors
# 7 num_oscar_winning_directors
POWER = [-2]
for power in POWER: 
    X_train = np.concatenate((X_train, X_train[:, 5][:, np.newaxis] ** power), axis=1)
    X_train = np.concatenate((X_train, X_train[:, 6][:, np.newaxis] ** power), axis=1)
    X_train = np.concatenate((X_train, X_train[:, 7][:, np.newaxis] ** power), axis=1)
"""

    
## Segmentation
low = 2
high = 1000
MASK = [X_train[:, 8] < low, 
        np.array([all(x) for x in zip(X_train[:,8]>low, X_train[:,8]<high)]),
        X_train[:, 8] > high]        
        
#discard = set([1, 7, 15, 17, 25, 28, 32, 38, 40, 41, 46, 54])
#keep = np.array(list(set(range(0,61)) - discard))
keep = np.arange(X_train.shape[1])
#keep = np.arange(50, 55)
#keep = np.array(range(37, 50))
#keep = [11]

rss = 0    
for i in range(len(MASK)):    
    mask = MASK[i]
    X_regress = X_train[mask,:][:,keep]
    y_regress = y_train[mask, np.newaxis]
    
    from sklearn import linear_model
    regr = linear_model.LinearRegression()
    regr.fit(X_regress, y_regress)
    print("Average squared residual: %.5f"
      % np.mean((regr.predict(X_regress) - y_regress) ** 2))
    rss += np.sum((regr.predict(X_regress) - y_regress) ** 2)
    
print("Overall:  %.5f" % (rss/1147))

"""
if len(keep) == 1 and i == 3:
    pl.scatter(X_regress, y_regress,  color='black')
    pl.plot(X_regress, regr.predict(X_regress), color='blue',
            linewidth=3)
    
    pl.xticks(())
    pl.yticks(())
    
    pl.show()
"""
"""

display_set = [11]
y_plot = y_train[mask, np.newaxis]
X_plot = X_train[mask,:][:,display_set]
from pandas.tools.plotting import scatter_matrix
from pandas import DataFrame
df = DataFrame(np.concatenate((y_plot, X_plot), axis=1))
scatter_matrix(df, alpha=0.2, figsize=(15, 15), diagonal='kde')
"""


"""
# do not use production budget to predict low revenue movies
lamb_prod = 4 # ols optimal at 92
mask_prod = X_train[:, 8] < lamb_prod
X_train[mask_prod, 11] = 0


lamb_wc = 4 # ols optimal at 5
mask_wc = X_train[:, 8] < lamb_wc
X_train[mask_wc, 14] = 0

lamb_ratings = 200 # ols 0.91716 at 200
mask_ratings = X_train[:, 8] < lamb_ratings
set_ratings = np.arange(50, 54)
for i in set_ratings:
    X_train[mask_ratings,i] = 0
    
lamb_genre = 1 # ols 0.92377 at 1
mask_genre = X_train[:, 8] > lamb_genre
set_genre = np.arange(15, 37)
for i in set_genre:
    X_train[mask_genre,i] = 0
       
lamb_genre = 50 # ols 0.93673 at 50
mask_genre = X_train[:, 8] > lamb_genre
set_genre = np.arange(55, 61)
for i in set_genre:
    X_train[mask_genre,i] = 0
"""


"""
lamb_screen = 1000 # ols optimal at 92
mask_screen = X_train[:, 8] < lamb_screen
X_train[mask_screen, 8] = 0

lamb_company = 500 # ols optimal at 5
mask_company = X_train[:, 8] > lamb_company
set_company = np.arange(37, 50)
for i in set_company:
    X_train[mask_company,i] = 0
"""
 # discard highgest_grossing_actors_present because missing: 1
# discard num_oscar_winning_directors: 7
# discard genres columns  under 6 counts: 15, 17, 25, 28, 32 
# discard companies columns under 1 counts: 38, 40, 41, 46
# discard ratings collumns under 2 counts: 54

