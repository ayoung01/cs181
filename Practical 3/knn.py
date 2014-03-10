# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 13:53:14 2014

@author: vincentli2010
"""


import numpy as np
import pylab as pl

from sklearn.preprocessing import StandardScaler

X = np.load(open('x_train3', 'rb'))
y = np.array(np.load(open('y_train', 'rb'))).flatten()

X_train, X_test = X[:2468], X[2468:]
y_train, y_test = y[:2468], y[2468:]

# Transformation
X = np.log(X + 1)
X = StandardScaler().fit_transform(X)


# weights='uniform',  p=2, 0.87686 +- 0.01346 for {'n_neighbors': 1}
# weights='uniform',  p=1, 0.88237 +- 0.01208 for {'n_neighbors': 1}
# weights='distance', p=2, 0.88334 +- 0.00912 for {'n_neighbors': 6 or 9}
# weights='distance', p=1, 0.88950 +- 0.00784 for {'n_neighbors': 9}

from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV

knn = KNeighborsClassifier(n_neighbors=5, weights='distance',
                           metric='minkowski', p=2)

tuned_parameters = [{'n_neighbors': np.arange(1, 20, 1)}]
knn_cv = GridSearchCV(knn, tuned_parameters,
                         cv=5, n_jobs=2, refit= False).fit(X, y)
for params, mean_score, scores in knn_cv.grid_scores_:
    if mean_score == knn_cv.best_score_:
        print ("KNN %.5f +- %.5f for %s" %
            (mean_score, scores.std(), params))


# shrink_threshold=0.51 0.394822006472
from sklearn.neighbors.nearest_centroid import NearestCentroid
print NearestCentroid(shrink_threshold=0.51).fit(X_train, y_train).score(X_test, y_test)