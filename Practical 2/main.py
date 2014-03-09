# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 14:47:08 2014

@author: vincentli2010
"""
"""


"""

##################################
##
## DATA IMPORT
##
##################################
"""
X_train.shape = (1147, n_feaures)
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


#topic_idx = np.load(open('topic_idx.npy', 'rb'))
#X_train = np.concatenate((np.load(open('feat.npy', 'rb')),
#                        np.load(open('/home/vincentli2010/Desktop/feat3.npy', 'rb'))[:,topic_idx]),
#                        axis=1)
#X_train = np.load(open('feat.npy', 'rb'))
#X_author = np.load(open('/home/vincentli2010/Desktop/feat4.npy', 'rb'))
#X_train = np.concatenate((X_train, X_author), axis=1)

#X = np.concatenate((np.load(open('feat.npy', 'rb')),
#                    np.load(open('/home/vincentli2010/Desktop/feat2.npy'))),axis=1)
#X_train = X
#X_test = X

"""
Segment 0:
main: 235282 ridge 4
main + joshi: 211286 pls 1
main + unigrams: 137748
* main + 11000 grams : 118019
main + 30000 grams filtered : 130301
main + gram_4std: 208389 ridge 0.9
main + gram_3std: 192133 pls 2
main + gram_2std: 177005 pls 1
main + gram_1std: 160951 pls 2
main + gram_0std: 147908 pls 2
main + gram_scale_4std: 219534 ridge 1
main + gram_scale_3std: 166291 pls 1
main + gram_scale_2std: 156360 pls 2
main + gram_scale_1std: 161962 pls 1
main + gram_scale_0std:152392 pls 2

Segment 1:
main: 5583528 ridge 0.5
main + joshi: 5842121 ridge 1
main + unigrams: 5485873
main + 11000 grams : 5188471
main + 30000 grams filtered: 6109536
main + gram_4std: 5459074 ridge 4.7
main + gram_3std:
main + gram_2std: 5409750 ridge 9
main + gram_1std: 5698353 ridge 12
* main + gram_0std: 5022290 ridge 0.9
main + gram_scale_4std: 5383745 ridge 4
main + gram_scale_3std: 5677077 ridge 5
main + gram_scale_2std: 5644106 ridge 8
main + gram_scale_1std: 6361560 ridge 2 only 7 variables selected
main + gram_scale_0std: 5862406 ridge 3

Best:
1875244
"""

import numpy as np
train_flag = True
import pickle

_, _, y_train, _ = pickle.load(open('features.p', 'rb'))

#X_train = np.load(open('/home/vincentli2010/Desktop/train_main.npy', 'rb'))
#X_train_grams = np.load(open('/home/vincentli2010/Desktop/train_grams.npy', 'rb'))
#X_train = np.concatenate((X_train_main, X_train_grams), axis=1)
X_train = np.load(open('train_joshi.npy', 'rb'))
#screen = np.sum(X_train,axis=0) > 10
#X_train = X_train[:, screen]
#X_test = np.load(open('/home/vincentli2010/Desktop/test_main.npy', 'rb'))
#X_test_grams = np.load(open('/home/vincentli2010/Desktop/test_grams.npy', 'rb'))
#X_test = np.concatenate((X_test_main, X_test_grams), axis=1)
X_test = np.load(open('test_joshi.npy', 'rb'))
#X_test = X_test[:, screen]

y_hat = np.empty((X_test.shape[0],))

n_jobs = -1
param_ridge_post= np.concatenate((#np.arange(0.01,0.1,0.01),
                                  np.arange(0.1,1,0.1),
                                  np.arange(1,10,1),))
rlasso_selection_threshold = 0
outfile = 'pred_joshi_cv20.csv'
##################################
##
## IMPUTATION
##
##################################
from sklearn.preprocessing import Imputer
imp_train = Imputer(missing_values= -1, strategy='mean', axis=0)
X_train = imp_train.fit_transform(X_train)

imp_test = Imputer(missing_values= -1, strategy='mean', axis=0)
X_test = imp_test.fit_transform(X_test)

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
        X_train = np.concatenate((X_train, X_train[:, 8][:, np.newaxis] ** power), axis=1)

    # production_budget trasnform by taking power and log
    POWER = [0.3]
    for power in POWER:
        X_train = np.concatenate((X_train, X_train[:, 11][:, np.newaxis] ** power), axis=1)
    X_train = np.concatenate((X_train, np.log(X_train[:, 11][:, np.newaxis])), axis=1)


    # number_of_screens
    POWER = [2]
    for power in POWER:
        X_test = np.concatenate((X_test, X_test[:, 8][:, np.newaxis] ** power), axis=1)

    # production_budget trasnform by taking power and log
    POWER = [0.3]
    for power in POWER:
        X_test = np.concatenate((X_test, X_test[:, 11][:, np.newaxis] ** power), axis=1)
    X_test = np.concatenate((X_test, np.log(X_test[:, 11][:, np.newaxis])), axis=1)

##################################
##
## Per Screen Prediction
##
##################################
per_screen_flag = False
if per_screen_flag:
    y_train = np.array([float(y)/float(X_train[i, 8]) for i, y in enumerate(y_train)])
    y_train = np.log(y_train)

##################################
##
## Segmentation
##
##################################
seg_flag = True
if seg_flag:
    l0 = 1000 #2151493 @ (1000)
    MASK = [X_train[:, 8] < l0,
            X_train[:, 8] > l0]
    for i, mask in enumerate(MASK):
        print "Train Segment %d\t%d" % (i, np.sum(mask!=0))

    print
    MASK_TEST = [X_test[:, 8] < l0,
                 X_test[:, 8] > l0]
    for i, mask in enumerate(MASK_TEST):
        print "Test Segment %d\t%d" % (i, np.sum(mask!=0))
    """
    MASK = [X_train[:, 8] > 0] #3937835

    l0 = 5 #2098825 @ (5, 1000)
    l1 = 1000
    keep = np.arange(X_train.shape[1])

    MASK = [X_train[:, 8] < l0,
            np.array([all(x) for x in zip(X_train[:, 8]>l0, X_train[:, 8]<l1)]),
            X_train[:, 8] > l1]

    l0 = 2 #2097856 @ (2, 10, 1000)
    l1 = 10
    l2 = 1000
    keep = np.arange(X_train.shape[1])

    MASK = [X_train[:, 8] < l0,
            np.array([all(x) for x in zip(X_train[:, 8]>l0, X_train[:, 8]<l1)]),
            np.array([all(x) for x in zip(X_train[:, 8]>l1, X_train[:, 8]<l2)]),
            X_train[:, 8] > l2]

    """
else:
    MASK = [np.array([True] * X_train.shape[0])]
    MASK_TEST = [np.array([True] * X_test.shape[0])]
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

    X = X_train[mask,:]
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
    X = X_train[mask,:]
    y = y_train[mask]
    X_test_masked = X_test[MASK_TEST[i],:]

    N_SEG.append(X.shape[0])

    # for segment 1, only use train_main features
    #if i == 1:
    #    X = X_train[mask,:][:, range(0, X_main.shape[1]) + \
    #        range(X_train.shape[1]-3, X_train.shape[1])]
    #    X_test_masked = X_test[MASK_TEST[i],:][:, range(0, X_main.shape[1]) + \
    #        range(X_train.shape[1]-3, X_train.shape[1])]

    # parameters search range

    #param_ridge_post = [330, 0.5] #p=24489
    #param_ridge_post = [3.7, 0.5] #p=303 2151493.01295

    # fit
    import linearall
    pre_pred = False
    model = linearall.LinearAll(cv=20, scoring = scorer,
                      n_jobs=n_jobs, refit=True, iid=False, pre_pred=pre_pred,
                      param_ridge_post=param_ridge_post,
                      rlasso_selection_threshold = rlasso_selection_threshold)
    model.fit(X, y)
    y_hat[MASK_TEST[i]] = model.predict(X_test_masked).flatten()

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
    if rlasso_selection_threshold == 0:
        print "Selected:\t%d/%d" % (np.sum(model.lasso_refit.coef_ != 0),X.shape[1])
    else:
        print "Selected:\t%d/%d" % (np.sum(model.rlasso.scores_ > rlasso_selection_threshold),X.shape[1])

    print "============ Post Variable Selection ==============="
    #print "OLS\t%.5f +- %.5f" % (-np.mean(model.ols_post), np.std(model.ols_post))

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
n_all, p_all = X_train.shape

ols_train_score = 0
#SELECTED_COEF = []
SELECTED = []
ols_post_score = 0
pls_post_score = 0
ridge_post_score = 0
for i, model in enumerate(MODEL):
    #ols_train_score += model.rss_ols_train
    if rlasso_selection_threshold == 0:
        SELECTED.append(np.sum(model.lasso_refit.coef_ != 0))
    else:
        SELECTED.append(np.sum(model.rlasso.scores_ > rlasso_selection_threshold))
    #SELECTED_COEF.append(model.lasso_refit.coef_)
    #ols_post_score += -np.mean(model.ols_post) * N_SEG[i]

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
#print "OLS\t%.5f" % ols_post_score
print "PLS\t%.5f" % pls_post_score
print "Ridge\t%.5f" % ridge_post_score

"""
# Visualize train fit
from pandas.tools.plotting import scatter_matrix
from pandas import DataFrame
df = DataFrame(np.concatenate((y_test[:,np.newaxis], y_hat[:,np.newaxis]), axis=1))
scatter_matrix(df, alpha=0.2, figsize=(15, 15), diagonal='kde')

print mean_absolute_error(np.exp(y_test), np.exp(y_hat))
"""

#Output predictions
y_out = np.exp(y_hat)
import util
test_ids = pickle.load(open('test_ids.p','rb'))
util.write_predictions(y_out, test_ids, outfile)





"""

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






"""

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
POWER = [-8, -7, -5, -4, -3, -2, -1, -0.1 ,0.1, 0.5, 2, 3, 5, 8, 8, 9]
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