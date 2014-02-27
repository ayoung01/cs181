# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 14:47:08 2014

@author: vincentli2010
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

import pickle
import numpy as np
import scipy.sparse as sp
discard, global_feat_dict, y_train, train_ids = pickle.load(open('features.p', 'rb'))
X_all = np.load(open('feat.npy', 'rb'))

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
# log y
y_train = np.log(y_train)

# number_of_screens
POWER = [2]
for power in POWER:
    X_all = np.concatenate((X_all, X_all[:, 8][:, np.newaxis] ** power), axis=1)

# production_budget trasnform by taking power and log
POWER = [0.3]
for power in POWER:
    X_all = np.concatenate((X_all, X_all[:, 11][:, np.newaxis] ** power), axis=1)
X_all = np.concatenate((X_all, np.log(X_all[:, 11][:, np.newaxis])), axis=1)

##################################
##
## Segmentation
##
##################################
seg_flag = True

if seg_flag:
    l0 = 1000 #2154673 @ (1000)
    MASK = [X_all[:, 8] < l0,
            X_all[:, 8] > l0]
    for i, mask in enumerate(MASK):
        print "Segment %d\t%d" % (i, np.sum(mask!=0))
else:
    MASK = [np.array([True] * X_all.shape[0])]


X_all = sp.csc_matrix(sp.hstack([sp.csc_matrix(X_all),
                   sp.csc_matrix(np.load(open('/home/vincentli2010/Desktop/feat3.npy', 'rb')))]))

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

"""
warning_show = False
if not warning_show:
    import warnings
    warnings.filterwarnings("ignore")
"""
keep = np.arange(X_all.shape[1])
mask = MASK[1]
X = X_all[mask,:][:,keep]
y = y_train[mask]


from sklearn import linear_model
model = linear_model.LassoCV(eps=0.001, n_alphas=100,
                             alphas=None, fit_intercept=True, normalize=False,
                             precompute='auto', max_iter=1, tol=0.0001,
                             copy_X=True, cv=2, verbose=False)
model.fit(X, y)
lasso_refit = linear_model.LassoLars(alpha=model.alpha_,
                            fit_intercept=True, normalize=True, precompute='auto',
                            max_iter=X.shape[1]+1000,
                            eps=2.2204460492503131e-16, copy_X=True,
                            fit_path=False)

lasso_refit= linear_model.Lasso(alpha=model.alpha_, fit_intercept=True, normalize=False,
      precompute='auto', copy_X=True, max_iter=1000, tol=0.0001,
      warm_start=False, positive=False)
lasso_refit.fit(X, y)
active = lasso_refit.coef_ != 0
X_selected = X[:, active[0,:]]

X_selected = X_selected.todense()


from sklearn.grid_search import GridSearchCV
from sklearn.cross_decomposition import PLSRegression
tuned_parameters = [{'n_components': range(1, 20)}]
pls = PLSRegression()
pls_cv = GridSearchCV(pls, tuned_parameters,
                        cv=5, scoring=scorer,
                        n_jobs=-1,
                        refit=False, iid=False)
pls_cv.fit(X_selected, y)
for params, mean_score, scores in pls_cv.grid_scores_:
    if mean_score == pls_cv.best_score_:
        print mean_score


tuned_parameters = [{'alpha': list(np.arange(0.1,3,0.1))}]
ridge = linear_model.Ridge(alpha = 1)
ridge_cv = GridSearchCV(ridge, tuned_parameters,
                        cv=5, scoring=scorer,
                        n_jobs=-1,
                        refit=False, iid=False)
ridge_cv.fit(X_selected, y)

for params, mean_score, scores in ridge_cv.grid_scores_:
    if mean_score == ridge_cv.best_score_:
        print mean_score








param_ridge_post= list(np.arange(0.1,1,0.1))
param_ridge_post.extend(list(np.arange(100,500,100)))






# fit
import linearall_sparse
pre_pred = False
model = linearall_sparse.LinearAll(cv=2, scoring = scorer,
                  n_jobs=1, refit=False, iid=False, pre_pred=pre_pred,
                  param_ridge_post=param_ridge_post)
model.fit(X, y)


##################################
##
## Print Summary
##
##################################
n_all, p_all = X_all.shape

ols_train_score = 0
SELECTED_COEF = []
SELECTED = []
ols_post_score = 0
pls_post_score = 0
ridge_post_score = 0
for i, model in enumerate(MODEL):
    #ols_train_score += model.rss_ols_train
    SELECTED.append(np.sum(model.lasso_refit.coef_ != 0))
    SELECTED_COEF.append(model.lasso_refit.coef_)
    ols_post_score += -np.mean(model.ols_post) * N_SEG[i]

    for params, mean_score, scores in model.pls_post.grid_scores_:
        if mean_score == model.pls_post.best_score_:
            pls_post_score += -mean_score * N_SEG[i]

    for params, mean_score, scores in model.ridge_post.grid_scores_:
        if mean_score == model.ridge_post.best_score_:
            ridge_post_score += -mean_score * N_SEG[i]

ols_train_score /= n_all
ols_post_score /= n_all
pls_post_score /= n_all
ridge_post_score /= n_all


print "\n\n\n\n\n==================================================="
print "=================== SUMMARY ======================="
print "==================================================="
print ("Train MSE(OLS):\t%.5f" % ols_train_score)


print "\n============== Variable Selection =================="
for i, mask in enumerate(MASK):
    print "Seg %d\tSelected:\t%d/%d" % (i, SELECTED[i], p_all)

print "\n============ Post Variable Selection ==============="
print "OLS\t%.5f" % ols_post_score
print "PLS\t%.5f" % pls_post_score
print "Ridge\t%.5f" % ridge_post_score

