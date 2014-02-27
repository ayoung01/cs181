# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 14:47:08 2014

@author: vincentli2010
"""
"""
This is split
"""

#import util

##################################
##
## DATA IMPORT
##
##################################
"""
X_all.shape = (1147, n_feaures)
8 number_of_screens
11 production_budget

14 review word count

55-60 sentiments 0.28247*** 58 -> 0.16029 # number of negative [0] sentences in review

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

61 np_release_dates
15-36 genres
37-49 companies
50-54 ratings
observations:
1. companies sharply divide high and low profile revenues
"""

import numpy as np
import pickle
discard, global_feat_dict, y_train, train_ids = pickle.load(open('features.p', 'rb'))
grams = np.load(open('/home/vincentli2010/Desktop/allgrams.npy', 'rb'))
#(1147, 33720)
divider = 20000
part1 = grams[:,range(20000)]
part2 = grams[:,range(20000, grams.shape[1])]
main = np.load(open('feat.npy', 'rb'))
X_1 = np.concatenate((main, part1),axis=1)
X_2 = np.concatenate((main, part2),axis=1)
X = [X_1, X_2]
collect = [set(), set()]

for j, X_all in enumerate(X):
    ##################################
    ##
    ## IMPUTATION
    ##
    ##################################
    from sklearn.preprocessing import Imputer
    imp = Imputer(missing_values= -1, strategy='mean', axis=0)
    X_all = imp.fit_transform(X_all)

    ##################################
    ##
    ## TRANSFORMATION
    ##
    ##################################
    transform_flag = True
    if transform_flag:
        # log y
        y_train = np.log(y_train)

    ##################################
    ##
    ## Segmentation
    ##
    ##################################
    seg_flag = True
    keep = np.arange(X_all.shape[1])
    if seg_flag:
        l0 = 1000 #2154673 @ (1000)
        MASK = [X_all[:, 8] < l0,
                X_all[:, 8] > l0]
        for i, mask in enumerate(MASK):
            print "Segment %d\t%d" % (i, np.sum(mask!=0))
        """
        MASK = [X_all[:, 8] > 0] #3937835

        l0 = 5 #2098825 @ (5, 1000)
        l1 = 1000
        keep = np.arange(X_all.shape[1])

        MASK = [X_all[:, 8] < l0,
                np.array([all(x) for x in zip(X_all[:,8]>l0, X_all[:,8]<l1)]),
                X_all[:, 8] > l1]

        l0 = 2 #2097856 @ (2, 10, 1000)
        l1 = 10
        l2 = 1000
        keep = np.arange(X_all.shape[1])

        MASK = [X_all[:, 8] < l0,
                np.array([all(x) for x in zip(X_all[:,8]>l0, X_all[:,8]<l1)]),
                np.array([all(x) for x in zip(X_all[:,8]>l1, X_all[:,8]<l2)]),
                X_all[:, 8] > l2]

        """
    else:
        MASK = [np.array([True] * X_all.shape[0])]

    ##################################
    ##
    ## Regression
    ##
    ##################################
    # scoring
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import make_scorer
    def scoring_f(y, y_hat):
         """
         CAUTION: err must be a LOSS function, the higher the score the better
         """
         err = -1 * mean_absolute_error(np.exp(y), np.exp(y_hat))
         #err = -1 * mean_squared_error(y, y_hat)
         return err
    scorer = make_scorer(scoring_f, greater_is_better=True)
    """
    CAUTION: greater_is_better=True => we are giving it a LOSS function
    """

    warning_show = False
    if not warning_show:
        import warnings
        warnings.filterwarnings("ignore")

    N_SEG = []
    MODEL = []
    for i, mask in enumerate(MASK):
        X = X_all[mask,:][:,keep]
        y = y_train[mask]
        N_SEG.append(X.shape[0])
        # parameters search range
        #param_ridge_post = list(np.arange(200,400,10))
        #param_ridge_post.append(0.5)
        param_ridge_post= np.concatenate((np.arange(0.1,1,0.1),np.arange(3,5,0.1)))
        #param_ridge_post = [330, 0.5] #p=24489
        #param_ridge_post = [3.7, 0.5] #p=303

        # fit
        from sklearn.linear_model import LassoLarsCV
        from sklearn import linear_model
        lasso_cv = LassoLarsCV(fit_intercept=True, normalize=True, precompute='auto',
                            max_iter=X.shape[1]+1000, max_n_alphas=X.shape[1]+1000,
                            eps= 2.2204460492503131e-16,copy_X=True,
                            cv=5, n_jobs=2)
        lasso_cv.fit(X, y)
        """
        normalize=True, lasso seems to be able to handle itself
        """

        lasso_refit = linear_model.LassoLars(alpha=lasso_cv.alpha_,
                            fit_intercept=True, normalize=True, precompute='auto',
                            max_iter=X.shape[1]+1000,
                            eps=2.2204460492503131e-16, copy_X=True,
                            fit_path=False)
        lasso_refit.fit(X, y)
        active = lasso_refit.coef_
        for i, x in enumerate(active[0]):
            if x != 0 and i > main.shape[1] - 1:
                collect[j].add(i)

collect_save = collect
idx1 = np.array(list(collect[0]))
idx2 = np.array(list(collect[1])) + divider
idx_filter = np.concatenate((idx1, idx2), axis=0)
X_save = np.concatenate((main, grams[:,idx_filter]),axis=1)
np.save(open('main_gram_filtered.npy', 'wb'), X_save)