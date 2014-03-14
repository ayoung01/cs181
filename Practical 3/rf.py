# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 13:53:14 2014

@author: vincentli2010
"""


# Load data
import numpy as np
y = np.array(np.load(open('y_train', 'rb'))).flatten()
X = np.load(open('x_train3', 'rb'))

# Transformation
from sklearn.preprocessing import StandardScaler
X = np.log(X + 1)
X = StandardScaler().fit_transform(X)

X_train = X
y_train = y
#X_train, X_test = X[:2468], X[2468:]
#y_train, y_test = y[:2468], y[2468:]

"""
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
"""

# n_estimators=300, max_features=20 oob_score = 0.901814646792
from sklearn.ensemble import ExtraTreesClassifier

erf = ExtraTreesClassifier(n_estimators=300, max_features=20,
                           bootstrap=True, oob_score=True,
                           criterion='gini',
                           max_depth=None, min_samples_split=2,
                           min_samples_leaf=1,
                           n_jobs=-1,
                           random_state=None, verbose=0, min_density=None,
                           compute_importances=None)
erf.fit(X_train, y_train)
print "oob\t%.4f" % (erf.oob_score_)

#pred_rf = erf.predict(X_test)
#print "pred_rf\t%.4f" % (erf.score(X_test, y_test))


# Output predictions
X_test = np.load(open('x_test', 'rb'))
from sklearn.preprocessing import StandardScaler
X_test = np.log(X_test + 1)
X_test = StandardScaler().fit_transform(X_test)
y_pred = erf.predict(X_test)



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
plt.title(mode#X_train, X_test = X[:2468], X[2468:]
#y_train, y_test = y[:2468], y[2468:]
l)
plt.show()
#plt.savefig('miss/' + model + '.png')
"""

"""
y_pred = erf.predict(X_test)
print erf.score(X_test, y_test)
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
cax = ax.matshow(miss, interpolation='nearest', vmin=0, vmax=18)
fig.colorbar(cax)
plt.xticks(range(len(target_names)), target_names, rotation=45)
plt.yticks(range(len(target_names)), target_names)
plt.title(model)
plt.show()
#plt.savefig('miss/' + model + '.png')


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

"""
#sleep = np.load(open('sleep', 'rb'))
#X = np.concatenate((X, sleep), axis=1)

#X_dll = np.load(open('dll_matrix', 'rb'))
#X = np.concatenate((X, X_dll), axis=1)

#X = np.load(open('dll_matrix', 'rb'))



#from sklearn.decomposition import PCA
#pca = PCA(n_components=20)
#pca.fit(X)
#print('explained variance: %f'
#      % np.sum(pca.explained_variance_ratio_))
#X = pca.transform(X)


"""
