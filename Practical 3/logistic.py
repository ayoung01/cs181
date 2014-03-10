# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 20:23:44 2014

@author: vincentli2010
"""

import numpy as np
import pylab as pl

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import l1_min_c

# x_train3 0.87622 +- 0.00969 for {'C': 1.4999999999999998}
# X_train2 0.86293 +- 0.00899 for {'C': 0.37384139481066314}
X_train = np.load(open('x_train3', 'rb'))
y_train = np.array(np.load(open('y_train', 'rb'))).flatten()


# Transformation
X_train = np.log(X_train + 1)
X_train = StandardScaler().fit_transform(X_train)

"""
from pandas.tools.plotting import scatter_matrix
from pandas import DataFrame
display_set = np.arange(10, 20, 1)
X_plot = X_train[:,display_set]
df = DataFrame(X_plot)
scatter_matrix(df, alpha=0.2, figsize=(15, 15), diagonal='kde')
"""


# CV on Train set
log_regr = LogisticRegression(penalty='l1', fit_intercept=True)

from sklearn.grid_search import GridSearchCV
#cs = l1_min_c(X_train, y_train, loss='log') * np.logspace(0, 4)
cs = np.arange(0.6, 2, 0.1)
tuned_parameters = [{'C': cs}]
log_cv = GridSearchCV(log_regr, tuned_parameters,
                         cv=5, n_jobs=3, refit= False)
log_cv.fit(X_train, y_train)

for params, mean_score, scores in log_cv.grid_scores_:
    if mean_score == log_cv.best_score_:
        print ("Logistic Regression\t%.5f +- %.5f for %s" %
            (mean_score, scores.std(), params))

"""
# Plot Regularization Paths
print("Computing regularization path ...")
clf = LogisticRegression(penalty='l1', fit_intercept=True)
coefs_ = []
for c in cs:
    clf.set_params(C=c)
    clf.fit(X_train, y_train)
    coefs_.append(clf.coef_.ravel().copy())
coefs_ = np.array(coefs_)
pl.plot(np.log10(cs), coefs_)
ymin, ymax = pl.ylim()
pl.xlabel('log(C)')
pl.ylabel('Coefficients')
pl.title('Logistic Regression Path')
pl.axis('tight')
pl.show()
"""