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

#poly C=35 degree=2 0.88950 +- 0.01272
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
tuned_parameters = [{'C': np.arange(20, 40, 3)}]
svm = SVC(kernel='poly', degree=2, coef0=1.0,
          cache_size=2000, max_iter=-1)
svm_cv = GridSearchCV(svm, tuned_parameters,
                         cv=5, n_jobs=-1, refit= False, verbose=1).fit(X, y)
for params, mean_score, scores in svm_cv.grid_scores_:
    if mean_score == svm_cv.best_score_:
        print ("SVM %.5f +- %.5f for %s" %
            (mean_score, scores.std(), params))



from sklearn.svm import SVC
svm_best = SVC(C=35, kernel='poly', degree=2, coef0=1.0,
          cache_size=2000, max_iter=-1).fit(X_train, y_train)
print "svm\t%f" % svm_best.score(X_test, y_test)
pred_svm = svm_best.fit(X_train, y_train).predict(X_test)



"""
model = 'svm'
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

## graph full miss classification matrix
miss = np.zeros((15,15), dtype=int)
for i in xrange(len(y_test)):
    if y_test[i] != y_pred[i]:
        miss[int(y_test[i]), int(y_pred[i])] += 1
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
plt.savefig('miss/' + model + 'full.png')
"""


