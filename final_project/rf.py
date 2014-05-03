# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 13:53:14 2014

Ghost classification

@author: vincentli2010
"""


# Load data
import numpy as np
#raw = np.genfromtxt('/home/vincentli2010/Desktop/ghost_train.csv', delimiter=',')[:10000]
#np.save(open('ghost_train.npy', 'wb'), raw)
raw = np.load(open('ghost_train.npy', 'rb'))
raw = raw[1:,:]
y_class = raw[:, 5]
#y_class = np.array([0 if y ==5 else 1 for y in y_class])
exclude_bad_ghosts = y_class != 5
y_class = y_class[exclude_bad_ghosts]
y_score = raw[:, 6]
y_score = y_score[exclude_bad_ghosts]

X = raw[:, 7:15]
#X = np.concatenate( (raw[:, 4:5], raw[:, 7:15]) , axis=1)
X = X[exclude_bad_ghosts,:]


X_train = X
y_train = y_class

# n_estimators=300, max_features=20 oob_score = 0.901814646792
from sklearn.ensemble import ExtraTreesClassifier
erf = ExtraTreesClassifier(n_estimators=40, max_features='auto',
                           bootstrap=True, oob_score=True,
                           criterion='gini',
                           max_depth=None, min_samples_split=2,
                           min_samples_leaf=1,
                           n_jobs=-1,
                           random_state=None, verbose=0, min_density=None,
                           compute_importances=None)
erf.fit(X_train, y_train)
print "PRE-selection oob\t%.4f" % (erf.oob_score_)

#pred_full = erf.predict(X_test)

feature_importance = erf.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())
print feature_importance


"""
from sklearn.metrics import mean_absolute_error
X_train = X[y_class == 3]
y_train = y_score[y_class == 3]
from sklearn import linear_model
clf0 = linear_model.LinearRegression()
clf0.fit(X_train, y_train)
pred = clf0.predict(X_train)
print mean_absolute_error(pred, y_train)
"""



#import pickle
#pickle.dump(erf, open('ghost_predictor.p', 'w'))


#from sklearn.externals import joblib
score_predictor = erf
#joblib.dump(ghost_predictor, 'ghost_predictor.pkl', compress=9)
#ghost_predictor = joblib.load('ghost_predictor.pkl')
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


"""
# Output predictions
y_pred = erf.predict(X_test)
ids = np.load(open('ids', 'rb'))
import util
util.write_predictions(y_pred, ids, 'predictions/rf_pc10.csv')
"""

"""
feature_names = np.load(open('names3', 'rb'))
# Plot feature importance
feature_importance = erf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(-feature_importance)
SELECTED = sorted_idx[:80]


X_train = X_train[:, SELECTED]
X_test = X_test[:, SELECTED]

# n_estimators=300, max_features=20 oob_score = 0.901814646792

from sklearn.ensemble import ExtraTreesClassifier
erf = ExtraTreesClassifier(n_estimators=300, max_features='auto',
                           bootstrap=True, oob_score=True,
                           criterion='gini',
                           max_depth=None, min_samples_split=2,
                           min_samples_leaf=1,
                           n_jobs=-1,
                           random_state=None, verbose=0, min_density=None,
                           compute_importances=None)
erf.fit(X_train, y_train)
print "POST-selection oob\t%.4f" % (erf.oob_score_)
#print "Test oob\t%.4f" % erf.score(X_test, y_test)
pred_selected = erf.predict(X_test)

# Output predictions
#y_pred = erf.predict(X_test)

from sklearn.metrics import accuracy_score
print accuracy_score(pred_full, pred_selected)

ids = np.load(open('ids', 'rb'))
import util
util.write_predictions(pred_selected, ids, 'predictions/erf_80var.csv')

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
