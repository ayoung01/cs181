# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 20:23:44 2014

@author: vincentli2010
"""

import numpy as np
import pylab as pl

from sklearn.preprocessing import StandardScaler
from sklearn.lda import LDA

X_train = np.load(open('x_train3', 'rb'))
y_train = np.array(np.load(open('y_train', 'rb'))).flatten()


# Transformation
X_train = np.log(X_train + 1)
X_train = StandardScaler().fit_transform(X_train)



from sklearn.linear_model import LogisticRegression
best_estimator = LogisticRegression(C = 1.5, penalty='l1', fit_intercept=True)
best_estimator.fit(X_train, y_train)
coef_l1_LR = best_estimator.coef_.ravel()
sparsity_l1_LR = np.mean(coef_l1_LR == 0) * 100
print sparsity_l1_LR
X_selected = best_estimator.transform(X_train, 8)

# QDA
from sklearn.grid_search import GridSearchCV
from sklearn.qda import QDA

cs = np.logspace(0, 3) * 1e-3
tuned_parameters = [{'reg_param': cs}]
qda = QDA(priors=None, reg_param=0.0)
qda_cv = GridSearchCV(qda, tuned_parameters,
                         cv=5, n_jobs=3, refit= False)
qda_cv.fit(X_selected, y_train)
for params, mean_score, scores in qda_cv.grid_scores_:
    if mean_score == qda_cv.best_score_:
        print ("QDA %.5f +- %.5f for %s" %
            (mean_score, scores.std(), params))

