# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 20:23:44 2014

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

X_train, X_test = X[:2468], X[2468:]
y_train, y_test = y[:2468], y[2468:]

from sklearn import ensemble

# x_train3 0.875404530744
original_params = {'n_estimators': 100, 'max_depth': 2, 'random_state': 1,
                   'min_samples_split': 2, 'learning_rate': 0.1, 'subsample': 0.5}

pl.figure()
params = dict(original_params)

clf = ensemble.GradientBoostingClassifier(**params)
clf.fit(X_train, y_train)

print clf.score(X_test, y_test)

color = 'orange'
label = 'learning_rate 0.1, subsample: 0.5'
# compute test set deviance
test_deviance = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
    # clf.loss_ assumes that y_test[i] in {0, 1}
    test_deviance[i] = clf.loss_(y_test, y_pred)
pl.plot((np.arange(test_deviance.shape[0]) + 1)[::5], test_deviance[::5],
        '-', color=color, label=label)

pl.legend(loc='upper left')
pl.xlabel('Boosting Iterations')
pl.ylabel('Test Set Deviance')

pl.show()

# Plot feature importance
feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
pl.subplot(1, 2, 2)
pl.barh(pos, feature_importance[sorted_idx], align='center')
#pl.yticks(pos, boston.feature_names[sorted_idx])
pl.xlabel('Relative Importance')
pl.title('Variable Importance')
pl.show()