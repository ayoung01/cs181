# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 20:23:44 2014

@author: vincentli2010
"""

import numpy as np

from sklearn.preprocessing import StandardScaler

X = np.load(open('x_train3', 'rb'))
y = np.array(np.load(open('y_train', 'rb'))).flatten()


# Transformation
X = np.log(X + 1)
X = StandardScaler().fit_transform(X)

X_train, X_test = X[:2468], X[2468:]
y_train, y_test = y[:2468], y[2468:]

# rbf C=100 gamma=0.005    0.886731391586
# poly C=20 degree=1   0.867313915858
# poly C=50 degree=2   0.888349514563
# poly C=50 degree=3   0.888349514563
# poly C=40 degree=4   0.888349514563


"""
from sklearn.svm import SVC
clf = SVC(C=50,
          kernel='poly', gamma=0.005, degree=2, coef0=1.0,
          cache_size=2000, max_iter=-1)
clf.fit(X_train, y_train)
print clf.score(X_test, y_test)
"""

#poly C=30 degree=2   0.88983 +- 0.01283
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
tuned_parameters = [{'C': np.arange(10, 100, 5)}]
svm = SVC(kernel='poly', degree=2, coef0=1.0,
          cache_size=2000, max_iter=-1)
svm_cv = GridSearchCV(svm, tuned_parameters,
                         cv=5, n_jobs=3, refit= False).fit(X, y)
for params, mean_score, scores in svm_cv.grid_scores_:
    if mean_score == svm_cv.best_score_:
        print ("QDA %.5f +- %.5f for %s" %
            (mean_score, scores.std(), params))





