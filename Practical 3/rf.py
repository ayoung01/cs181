# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 13:53:14 2014

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

#n_estimators=200, max_features=10 oob_score = 0.901490602722 valid: 0.891585760518
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200, max_features=10,
                            criterion='gini',
                            max_depth=None, # full grown-trees
                            oob_score=True, n_jobs=-1,
                            min_density=None, compute_importances=None)
rf.fit(X_train, y_train)
print rf.oob_score_

#rf.fit(X_train, y_train)
#print rf.score(X_test, y_test)


# n_estimators=500, max_features=10 oob_score = 0.901490602722 valid: 0.901294498382
from sklearn.ensemble import ExtraTreesClassifier
erf = ExtraTreesClassifier(n_estimators=500, max_features=10,
                           bootstrap=True, oob_score=True,
                           criterion='gini',
                           max_depth=None, min_samples_split=2,
                           min_samples_leaf=1,
                           n_jobs=-1,
                           random_state=None, verbose=0, min_density=None,
                           compute_importances=None)
erf.fit(X_train, y_train)
print erf.oob_score_

#erf.fit(X_train, y_train)
#print erf.score(X_test, y_test)


"""
feature_names = np.load(open('names3', 'rb'))
import pylab as pl
# Plot feature importance
feature_importance = rf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

discard_bottom = 60
pos_plot = pos[discard_bottom:]
fi_plot = feature_importance[sorted_idx][discard_bottom:]
names_plot = feature_names[sorted_idx][discard_bottom:]
pl.subplot(1, 1, 1)
pl.barh(pos_plot, fi_plot, align='center')
pl.yticks(pos_plot, names_plot)
pl.xlabel('Relative Importance')
pl.title('Variable Importance RF')
pl.show()
"""



y_pred = rf.predict(X_test)
model = 'RF'

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
cax = ax.matshow(miss, interpolation='nearest')
fig.colorbar(cax)
plt.xticks(range(len(target_names)), target_names, rotation=45)
plt.yticks(range(len(target_names)), target_names)
plt.title(model)
plt.show()
plt.savefig('miss/' + model + '.png')


y_pred = erf.predict(X_test)
model = 'ERF'

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
cax = ax.matshow(miss, interpolation='nearest')
fig.colorbar(cax)
plt.xticks(range(len(target_names)), target_names, rotation=45)
plt.yticks(range(len(target_names)), target_names)
plt.title(model)
plt.show()
plt.savefig('miss/' + model + '.png')

