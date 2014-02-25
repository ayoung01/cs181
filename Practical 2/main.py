# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 14:47:08 2014

@author: vincentli2010
"""

import numpy as np
#import util

"""
X_train.shape = (1147, n_feaures)
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

##################################
##
## IMPUTATION
##
##################################
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values= -1, strategy='mean', axis=0)
X_train = imp.fit_transform(X_train)


##################################
##
## TRANSFORMATION
##
##################################

# log y
y_train = np.log(y_train)

# production_budget trasnform by taking power
POWER = [0.3]
for power in POWER:
    X_train = np.concatenate((X_train, X_train[:, 11][:, np.newaxis] ** power), axis=1)

# number_of_screens
POWER = [2]
for power in POWER:
    X_train = np.concatenate((X_train, X_train[:, 8][:, np.newaxis] ** power), axis=1)

##################################
##
## Basis Expansion
##
##################################
# production_budget trasnform by taking power
POWER = [-0.3, 0.1, 0.5, 1.5]
for power in POWER:
    X_train = np.concatenate((X_train, X_train[:, 11][:, np.newaxis] ** power), axis=1)
X_train = np.concatenate((X_train, np.log(X_train[:, 11][:, np.newaxis])), axis=1)

# number_of_screens
POWER = [-2, -1, 0.1, 0.5, 1.5]
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



##################################
##
## Segmentation
##
##################################
low = 2
high = 1000
MASK = [X_train[:, 8] < low,
        np.array([all(x) for x in zip(X_train[:,8]>low, X_train[:,8]<high)]),
        X_train[:, 8] > high]

keep = np.arange(X_train.shape[1])

mask = MASK[2]
X = X_train[mask,:][:,keep]
y = y_train[mask]



##################################
##
## Regression
##
##################################


import linearall
models = linearall.LinearAll(cv=20, scoring = 'mean_squared_error',
                  n_jobs=-1, refit=False, iid=False)
models.fit(X, y)

##################################
##
## Print shit
##
##################################
print "==================================================="
print "===================== SUMMARY ====================="
print "==================================================="
print ("Train OLS:\t%.5f" % (models.rss_ols_train/X.shape[0]))

print "\n============ Pre Variable Selection ==============="
print "OLS:\t%.5f +- %.5f" % (-np.mean(models.ols_pre), np.std(models.ols_pre))

#print("Grid scores on development set:")
#for params, mean_score, scores in models.pls_pre..grid_scores_:
#    print("%0.3f (+/-%0.03f) for %r"
#          % (mean_score, scores.std() / 2, params))
for params, mean_score, scores in models.pls_pre.grid_scores_:
    if mean_score == models.pls_pre.best_score_:
        print ("PLS\t%.5f +- %.5f for %s" %
            (-mean_score, scores.std(), params))

#print("Grid scores on development set:")
#for params, mean_score, scores in models.ridge_pre.grid_scores_:
#    print("%0.3f (+/-%0.03f) for %r"
#          % (mean_score, scores.std() / 2, params))
for params, mean_score, scores in models.ridge_pre.grid_scores_:
    if mean_score == models.ridge_pre.best_score_:
        print ("Ridge\t%.5f +- %.5f for %s" %
            (-mean_score, scores.std(), params))

print "\n============== Variable Selection =================="
print "Selected:\t%d/%d" % (np.sum(models.lasso_refit.coef_ != 0),X.shape[1])

print "\n============ Post Variable Selection ==============="
print "OLS:\t%.5f +- %.5f" % (-np.mean(models.ols_post), np.std(models.ols_post))

#print("Grid scores on development set:")
#for params, mean_score, scores in models.pls_post.grid_scores_:
#    print("%0.3f (+/-%0.03f) for %r"
#          % (mean_score, scores.std() / 2, params))
for params, mean_score, scores in models.pls_post.grid_scores_:
    if mean_score == models.pls_post.best_score_:
        print ("PLS\t%.5f +- %.5f for %s" %
            (-mean_score, scores.std(), params))

#print("Grid scores on development set:")
#for params, mean_score, scores in models.ridge_post.grid_scores_:
#    print("%0.3f (+/-%0.03f) for %r"
#          % (mean_score, scores.std() / 2, params))
for params, mean_score, scores in models.ridge_post.grid_scores_:
    if mean_score == models.ridge_post.best_score_:
        print ("Ridge\t%.5f +- %.5f for %s" %
            (-mean_score, scores.std(), params))










"""
mask = MASK[2]
X_regress = X_train[mask,:][:,keep]
y_regress = y_train[mask]

##################################
##
## Model Selection
##
##################################

### OLS Train Scores
rss = 0
for i in range(len(MASK)):
    mask = MASK[i]
    X_regress = X_train[mask,:][:,keep]
    y_regress = y_train[mask]

    from sklearn import linear_model
    regr = linear_model.LinearRegression(fit_intercept=True, normalize=False)
    regr.fit(X_regress, y_regress)
    print("Average squared residual: %.5f"
      % np.mean((regr.predict(X_regress) - y_regress) ** 2))
    rss += np.sum((regr.predict(X_regress) - y_regress) ** 2)

print("Overall:  %.5f" % (rss/1147))


print "Pre Variable Selection"
# validation scores for OLS
from sklearn import cross_validation
regr_final = linear_model.LinearRegression()
scores = -1 * cross_validation.cross_val_score(
        regr_final, X_regress, y_regress, cv=20, scoring = 'mean_squared_error')
#pl.clf()
#pl.plot(scores)
print "OLS: %.5f +- %.5f" % (np.mean(scores), np.std(scores))


from sklearn.grid_search import GridSearchCV
tuned_parameters = [{'n_components': range(1, 20)}]
from sklearn.cross_decomposition import PLSRegression
pls = PLSRegression()
clf = GridSearchCV(pls, tuned_parameters, cv=20, scoring='mean_squared_error',
                   n_jobs = 1)

clf.fit(X_regress, y_regress)
#print("Grid scores on development set:")
#for params, mean_score, scores in clf.grid_scores_:
#    print("%0.3f (+/-%0.03f) for %r"
#          % (mean_score, scores.std() / 2, params))
for params, mean_score, scores in clf.grid_scores_:
    if mean_score == clf.best_score_:
        print ("PLS Best: %.5f +- %.5f for %s" %
            (-mean_score, scores.std(), params))



alphas = list(np.arange(1e9,2e9,1e8))
tuned_parameters = [{'alpha': alphas}]
ridge = linear_model.Ridge(alpha = 1)
search = GridSearchCV(ridge, tuned_parameters, cv=20, scoring='mean_squared_error')
search.fit(X_regress, y_regress)
#print("Grid scores on development set:")
#for params, mean_score, scores in search.grid_scores_:
#    print("%0.3f (+/-%0.03f) for %r"
#          % (mean_score, scores.std() / 2, params))
for params, mean_score, scores in search.grid_scores_:
    if mean_score == search.best_score_:
        print ("Ridge Best: %.5f +- %.5f for %s" %
            (-mean_score, scores.std(), params))


print "Variable Selection"
import time
from sklearn.linear_model import LassoLarsCV
import pylab as pl

print("Computing regularization path using the Lars lasso...")
t1 = time.time()
model = LassoLarsCV(fit_intercept=True, normalize=True, precompute='auto',
                            max_iter=X_regress.shape[1]+1000, max_n_alphas=X_regress.shape[1]+1000,
                            eps= 2.2204460492503131e-16,copy_X=True,
                            cv=20, n_jobs=-1).fit(X_regress, y_regress)
t_lasso_lars_cv = time.time() - t1

# Display results
m_log_alphas = -np.log10(model.cv_alphas_)



clf = linear_model.LassoLars(alpha=model.alpha_, fit_path=False)
clf.fit(X_regress, y_regress)
active = clf.coef_ != 0
X_regress= X_regress[:, active[0,:]]


# validation scores for OLS
from sklearn import cross_validation
regr_final = linear_model.LinearRegression()
scores = -1 * cross_validation.cross_val_score(
        regr_final, X_regress, y_regress, cv=20, scoring = 'mean_squared_error')
#pl.clf()
#pl.plot(scores)
print "OLS: %.5f +- %.5f" % (np.mean(scores), np.std(scores))


from sklearn.grid_search import GridSearchCV
tuned_parameters = [{'n_components': range(1, X_regress.shape[1])}]
from sklearn.cross_decomposition import PLSRegression
pls = PLSRegression(n_components=1)
clf = GridSearchCV(pls, tuned_parameters, cv=20, scoring='mean_squared_error')
clf.fit(X_regress, y_regress)
#print("Grid scores on development set:")
#for params, mean_score, scores in clf.grid_scores_:
#    print("%0.3f (+/-%0.03f) for %r"
#          % (mean_score, scores.std() / 2, params))
for params, mean_score, scores in clf.grid_scores_:
    if mean_score == clf.best_score_:
        print ("PLS Best: %.5f +- %.5f for %s" %
            (-mean_score, scores.std(), params))


alphas = list(np.arange(1,3,0.1))
tuned_parameters = [{'alpha': alphas}]
ridge = linear_model.Ridge(alpha = 1)
search = GridSearchCV(ridge, tuned_parameters, cv=20, scoring='mean_squared_error')
search.fit(X_regress, y_regress)
#print("Grid scores on development set:")
#for params, mean_score, scores in clf.grid_scores_:
#    print("%0.3f (+/-%0.03f) for %r"
#          % (mean_score, scores.std() / 2, params))
for params, mean_score, scores in search.grid_scores_:
    if mean_score == search.best_score_:
        print ("Ridge Best: %.5f +- %.5f for %s" %
            (-mean_score, scores.std(), params))
"""