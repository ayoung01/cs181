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
train_flag = True
if train_flag:
    import pickle
    discard, global_feat_dict, y_train, train_ids = pickle.load(open('features.p', 'rb'))
    X_all = np.concatenate((np.load(open('feat.npy', 'rb')),
                            np.load(open('/home/vincentli2010/Desktop/feat3.npy', 'rb'))[:,range(20000,33720)]),
                            axis=1)
else:
    X_all = np.load(open('feat_test.npy', 'rb'))

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
## Per Screen Prediction
##
##################################
per_screen_flag = False
if per_screen_flag:
    y_train = np.array([float(y)/float(X_all[i,8]) for i, y in enumerate(y_train)])
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
## Visualization
##
##################################
v_flag = False
if v_flag:
    from pandas.tools.plotting import scatter_matrix
    from pandas import DataFrame

    display_set =[87]
    mask = MASK[1]

    X = X_all[mask,:][:,keep]
    X_plot = X[:,display_set]


    if train_flag:
        y = y_train[mask]
        y_plot = y[:, np.newaxis]
        df = DataFrame(np.concatenate((y_plot, X_plot), axis=1))
    else:
        df = DataFrame(X_plot)
    scatter_matrix(df, alpha=0.2, figsize=(15, 15), diagonal='kde')


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
    #param_ridge_post= np.concatenate((np.arange(0.5,0.6,0.1),np.arange(3,5,0.1)))
    param_ridge_post = [330, 0.5] #p=24489
    #param_ridge_post = [3.7, 0.5] #p=303

    # fit
    import linearall
    pre_pred = False
    model = linearall.LinearAll(cv=5, scoring = scorer,
                      n_jobs=1, refit=False, iid=False, pre_pred=pre_pred,
                      param_ridge_post=param_ridge_post)
    model.fit(X, y)
    MODEL.append(model)

    # print for each segment
    print "\n\n================= SEGMENT %d =====================" % i
    #print ("Train MSE(OLS):\t%.5f" % (model.rss_ols_train/X.shape[0]))


    if pre_pred:
        print "============ Pre Variable Selection ==============="
        print "OLS\t%.5f +- %.5f" % (-np.mean(model.ols_pre), np.std(model.ols_pre))

        #print("Grid scores on development set:")
        #for params, mean_score, scores in models.pls_pre..grid_scores_:
        #    print("%0.3f (+/-%0.03f) for %r"
        #          % (mean_score, scores.std() / 2, params))
        for params, mean_score, scores in model.pls_pre.grid_scores_:
            if mean_score == model.pls_pre.best_score_:
                print ("PLS\t%.5f +- %.5f for %s" %
                    (-mean_score, scores.std(), params))

        #print("Grid scores on development set:")
        #for params, mean_score, scores in models.ridge_pre.grid_scores_:
        #    print("%0.3f (+/-%0.03f) for %r"
        #          % (mean_score, scores.std() / 2, params))
        for params, mean_score, scores in model.ridge_pre.grid_scores_:
            if mean_score == model.ridge_pre.best_score_:
                print ("Ridge\t%.5f +- %.5f for %s" %
                    (-mean_score, scores.std(), params))

    print "============== Variable Selection =================="
    print "Selected:\t%d/%d" % (np.sum(model.lasso_refit.coef_ != 0),X.shape[1])


    print "============ Post Variable Selection ==============="
    print "OLS\t%.5f +- %.5f" % (-np.mean(model.ols_post), np.std(model.ols_post))

    #print("Grid scores on development set:")
    #for params, mean_score, scores in models.pls_post.grid_scores_:
    #    print("%0.3f (+/-%0.03f) for %r"
    #          % (mean_score, scores.std() / 2, params))
    for params, mean_score, scores in model.pls_post.grid_scores_:
        if mean_score == model.pls_post.best_score_:
            print ("PLS\t%.5f +- %.5f for %s" %
                (-mean_score, scores.std(), params))

    #print("Grid scores on development set:")
    #for params, mean_score, scores in models.ridge_post.grid_scores_:
    #    print("%0.3f (+/-%0.03f) for %r"
    #          % (mean_score, scores.std() / 2, params))
    for params, mean_score, scores in model.ridge_post.grid_scores_:
        if mean_score == model.ridge_post.best_score_:
            print ("Ridge\t%.5f +- %.5f for %s" %
                (-mean_score, scores.std(), params))

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


#collect = set()
for i, x in enumerate(SELECTED_COEF[1][0]):
    if x != 0:
        if i > 300:
            if i != 14022:
                if i != 14021:
                    collect.add(i + 20000)
collect = np.array(collect)
collect -= 1
np.save('topic_idx.npy', collect)
"""
mask = MASK[2]
X_regress = X_all[mask,:][:,keep]
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
    X_regress = X_all[mask,:][:,keep]
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






"""

##################################
##
## Basis Expansion
##
##################################


# production_budget trasnform by taking power
POWER = [-0.3, 0.1, 0.5, 1.5]
for power in POWER:
    X_all = np.concatenate((X_all, X_all[:, 11][:, np.newaxis] ** power), axis=1)
X_all = np.concatenate((X_all, np.log(X_all[:, 11][:, np.newaxis])), axis=1)

# number_of_screens
POWER = [-2, -1, 0.1, 0.5, 1.5]
for power in POWER:
    X_all = np.concatenate((X_all, X_all[:, 8][:, np.newaxis] ** power), axis=1)
X_all = np.concatenate((X_all, np.log(X_all[:, 8][:, np.newaxis])), axis=1)

#14 review word count
POWER = [-2, -1, 0.1, 0.5, 2, 3]
for power in POWER:
    X_all = np.concatenate((X_all, X_all[:, 14][:, np.newaxis] ** power), axis=1)
X_all = np.concatenate((X_all, np.log(X_all[:, 14][:, np.newaxis])), axis=1)

#sentiments: avg, numpos, numneg
POWER = [-8, -7, -5, -4, -3, -2, -1, -0.1 ,0.1, 0.5, 2, 3, 5, 7, 8, 9]
for i in [55, 57, 58]:
    for power in POWER:
        X_all = np.concatenate((X_all, (X_all[:, i][:, np.newaxis] + 1) ** power), axis=1)
    X_all = np.concatenate((X_all, np.log(X_all[:, i][:, np.newaxis] + 1)), axis=1)

# running time
POWER = [-2, -1, 0.1, 0.5, 2]
for power in POWER:
    X_all = np.concatenate((X_all, X_all[:, 12][:, np.newaxis] ** power), axis=1)
X_all = np.concatenate((X_all, np.log(X_all[:, 12][:, np.newaxis])), axis=1)


# np_release_dates
POWER = [-2, -1, 0.1, 0.5, 2, 3]
for power in POWER:
    X_all = np.concatenate((X_all, (X_all[:, 61][:, np.newaxis] + 1) ** power), axis=1)
X_all = np.concatenate((X_all, np.log(X_all[:, 61][:, np.newaxis] + 1)), axis=1)
"""