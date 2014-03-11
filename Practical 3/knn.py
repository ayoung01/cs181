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


#NC 0.61342 +- 0.05203 for {'shrink_threshold': 0.0012067926406393286}
from sklearn.neighbors.nearest_centroid import NearestCentroid
tuned_parameters = [{'shrink_threshold': np.logspace(0,4) * 0.001}]
nc = NearestCentroid(shrink_threshold=0.51)
nc_cv = GridSearchCV(nc, tuned_parameters,
                         cv=5, n_jobs=-1, refit= False).fit(X, y)
for params, mean_score, scores in nc_cv.grid_scores_:
    if mean_score == nc_cv.best_score_:
        print ("NC %.5f +- %.5f for %s" %
            (mean_score, scores.std(), params))




import numpy as np
import pylab as pl

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import l1_min_c

X = np.load(open('x_train3', 'rb'))
y = np.array(np.load(open('y_train', 'rb'))).flatten()

# Transformation
X = np.log(X + 1)
X = StandardScaler().fit_transform(X)

X_train, X_test = X[:2468], X[2468:]
y_train, y_test = y[:2468], y[2468:]

knn_best = KNeighborsClassifier(n_neighbors=9, weights='distance',
                           metric='minkowski', p=1)
y_pred = knn_best.fit(X_train, y_train).predict(X_test)
model = 'knn'

miss = np.zeros((15,15), dtype=int)
for i in xrange(len(y_test)):
    if y_test[i] != y_pred[i]:
        if y_test[i] < y_pred[i]:
            miss[int(y_test[i]), int(y_pred[i])] += 1
        else:
            miss[int(y_pred[i]), int(y_test[i])] += 1
import matplotlib.pyplot as plt
target_names =  ['Agent', 'AutoRun',
                 'FraudLoad', 'FraudPack',
                 'Hupigon', 'Krap',
                 'Lipler', 'Magania',
                 'None', 'Poison',
                 'Swizzor', 'Tdss',
                 'VB', 'Virut', 'Zbot']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(miss, interpolation='nearest', vmin=0, vmax=18)
fig.colorbar(cax)
plt.xticks(range(len(target_names)), target_names, rotation=45)
plt.yticks(range(len(target_names)), target_names)
plt.title(model)
plt.show()
plt.savefig('miss/' + model + '.png')

