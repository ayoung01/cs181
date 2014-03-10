# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 20:23:44 2014

@author: vincentli2010
"""

import numpy as np
import pylab as pl

from sklearn.preprocessing import StandardScaler

X = np.load(open('x_train2', 'rb'))
y = np.array(np.load(open('y_train', 'rb'))).flatten()


# Transformation
X = np.log(X + 1)
X = StandardScaler().fit_transform(X)

X_train, X_test = X[:2468], X[2468:]
y_train, y_test = y[:2468], y[2468:]

from sklearn import ensemble

original_params = {'n_estimators': 1000, 'max_depth': 2, 'random_state': 1,
                   'min_samples_split': 5}

pl.figure()

#from sklearn.metrics import accuracy_score
YPRED = []

for label, color, setting in [('No shrinkage', 'orange',
                               {'learning_rate': 1.0, 'subsample': 1.0}),
                              ('learning_rate=0.1', 'turquoise',
                               {'learning_rate': 0.1, 'subsample': 1.0}),
                              ('subsample=0.5', 'blue',
                               {'learning_rate': 1.0, 'subsample': 0.5}),
                              ('learning_rate=0.1, subsample=0.5', 'gray',
                               {'learning_rate': 0.1, 'subsample': 0.5}),
                              ('learning_rate=0.1, max_features=2', 'magenta',
                               {'learning_rate': 0.1, 'max_features': 2})]:
    print label
    params = dict(original_params)
    params.update(setting)

    clf = ensemble.GradientBoostingClassifier(**params)
    clf.fit(X_train, y_train)

    # compute test set deviance
    test_deviance = np.zeros((params['n_estimators'],), dtype=np.float64)

    for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
        # clf.loss_ assumes that y_test[i] in {0, 1}
        test_deviance[i] = clf.loss_(y_test, y_pred)
        YPRED.append(y_pred)
    pl.plot((np.arange(test_deviance.shape[0]) + 1)[::5], test_deviance[::5],
            '-', color=color, label=label)

pl.legend(loc='upper left')
pl.xlabel('Boosting Iterations')
pl.ylabel('Test Set Deviance')

pl.show()