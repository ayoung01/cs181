# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 20:23:44 2014

@author: vincentli2010
"""

import numpy as np
import pylab as pl

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import l1_min_c

X = np.load(open('x_train3', 'rb'))
#X_dll = np.load(open('dll_matrix', 'rb'))
#X = np.concatenate((X, X_dll), axis=1)
y = np.array(np.load(open('y_train', 'rb'))).flatten()

# Transformation
X = np.log(X + 1)
scaler = StandardScaler()
X = scaler.fit_transform(X)

#X_train, X_test = X[:2468], X[2468:]
#y_train, y_test = y[:2468], y[2468:]

X_train = X
y_train = y


from sklearn.decomposition import PCA
pca = PCA(n_components=100)
X_train = pca.fit(X_train).transform(X_train)
print('explained variance: %f'
      % np.sum(pca.explained_variance_ratio_))


"""
from pandas.tools.plotting import scatter_matrix
from pandas import DataFrame
display_set = np.arange(10, 20, 1)
X_plot = X_train[:,display_set]
df = DataFrame(X_plot)
scatter_matrix(df, alpha=0.2, figsize=(15, 15), diagonal='kde')
"""


# CV on Train set
log_regr = LogisticRegression(penalty='l1', fit_intercept=True)

from sklearn.grid_search import GridSearchCV
#cs = l1_min_c(X_train, y_train, loss='log') * np.logspace(0, 3)
#cs = np.arange(0.6, 2, 0.1)
#cs = np.logspace(0, 4) * 0.1
cs = [0.1, 1, 5, 10]
tuned_parameters = [{'C': cs}]
log_cv = GridSearchCV(log_regr, tuned_parameters,
                         cv=5, n_jobs=3, refit= False)
log_cv.fit(X_train, y_train)

for params, mean_score, scores in log_cv.grid_scores_:
    if mean_score == log_cv.best_score_:
        print ("Logistic Regression\t%.5f +- %.5f for %s" %
            (mean_score, scores.std(), params))


# L1 Sparsity 0.87622 +- 0.00969 for {'C': 1.4999999999999998}
best_estimator = LogisticRegression(C = 1.5, penalty='l1', fit_intercept=True)
best_estimator.fit(X_train, y_train)
coef_l1_LR = best_estimator.coef_.ravel()
sparsity_l1_LR = np.mean(coef_l1_LR == 0) * 100
print sparsity_l1_LR

X_selected = best_estimator.transform(X_train, 4)
spar = LogisticRegression(C = 10000, penalty='l1', fit_intercept=True)
spar.fit(X_selected, y_train)
coef_l1_LR = spar.coef_.ravel()
sparsity_l1_LR = np.mean(coef_l1_LR == 0) * 100
print sparsity_l1_LR

# L1 + L2 0.86909 +- 0.01231 for {'C': 10.985411419875584}
log_regr = LogisticRegression(penalty='l2', fit_intercept=True)

from sklearn.grid_search import GridSearchCV
#cs = l1_min_c(X_train, y_train, loss='log') * np.logspace(0, 4)
#cs = np.arange(0.6, 2, 0.1)
cs = np.logspace(0, 4) * 0.1
tuned_parameters = [{'C': cs}]
log_cv = GridSearchCV(log_regr, tuned_parameters,
                         cv=2, n_jobs=3, refit= False)
log_cv.fit(X_selected, y_train)

for params, mean_score, scores in log_cv.grid_scores_:
    if mean_score == log_cv.best_score_:
        print ("Logistic Regression\t%.5f +- %.5f for %s" %
            (mean_score, scores.std(), params))



# L2 Sparsity 0.86811 +- 0.01199 for {'C': 10.985411419875584}
best_estimator2 = LogisticRegression(C = 10.985411419875584, penalty='l2', fit_intercept=True)
best_estimator2.fit(X_train, y_train)
coef_l2_LR = best_estimator2.coef_.ravel()
sparsity_l2_LR = np.mean(coef_l2_LR == 0) * 100
print sparsity_l2_LR




import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

X = np.load(open('x_train3', 'rb'))
y = np.array(np.load(open('y_train', 'rb'))).flatten()

# Transformation
X = np.log(X + 1)
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test = X[:2468], X[2468:]
y_train, y_test = y[:2468], y[2468:]


# final model
l1_best = LogisticRegression(C= 1.5, penalty='l1', fit_intercept=True).fit(X_train, y_train)
print "logisticl1\t%.4f" % l1_best.score(X_test, y_test)
pred_lg = l1_best.predict(X_test)




# Output predictions
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

X = np.load(open('x_train3', 'rb'))
y = np.array(np.load(open('y_train', 'rb'))).flatten()

# Transformation
X = np.log(X + 1)
scaler = StandardScaler()
X = scaler.fit_transform(X)

l1_best = LogisticRegression(C= 1.5, penalty='l1', fit_intercept=True)
l1_best.fit(X, y)
print l1_best.score(X, y)

X_test = np.load(open('x_test', 'rb'))
from sklearn.preprocessing import StandardScaler
X_test = np.log(X_test + 1)
X_test = StandardScaler().fit_transform(X_test)
y_pred = l1_best.predict(X_test)


ids = np.load(open('ids', 'rb'))
import util
util.write_predictions(y_pred, ids, 'predictions/logisticl1.csv')







"""
#### Inspect misclassifications

import pickle as pickle
name_to_idx = pickle.load(open('names3', 'rb'))
malware_to_idx = pickle.load(open('malware', 'rb'))

## Inspect properties of misclassifications cases
# transform X into original scale for easier inspection
X0 = scaler.inverse_transform(X, copy=True)
X0 = np.exp(X0) - 1
X_test0 = X0[2468:]

i_list  = []
for i in xrange(len(y_pred)):
    if y_pred[i] == malware_to_idx['Virut'] and y_test[i] == malware_to_idx['None']:
        print 'i:%d \t%d ' % (i, X_test0[i,name_to_idx['file size']])
        i_list.append(i)


from sklearn.metrics import accuracy_score
accuracy_score(y_pred, y_test)
"""

"""
convert = []
for i, y in enumerate(y_pred):
    if y == malware_to_idx['None'] and \
            int(X_test0[i,name_to_idx['create_mutex']]) == 1:
        y_pred[i] = malware_to_idx['Virut']
        print 'none to virut %d' % i
        convert.append(i)
    elif y == malware_to_idx['Virut'] and \
            int(X_test0[i,name_to_idx['create_mutex']]) != 1:
        y_pred[i] = malware_to_idx['None']
        print 'virut to none: %d' % i
        convert.append(i)
"""


"""
## graph upper trianglar miss classification matrix
model = 'Logisticl1'
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

"""
# Plot Regularization Paths
print("Computing regularization path ...")
clf = LogisticRegression(penalty='l1', fit_intercept=True)
coefs_ = []
for c in cs:
    clf.set_params(C=c)
    clf.fit(X_train, y_train)
    coefs_.append(clf.coef_.ravel().copy())
coefs_ = np.array(coefs_)
pl.plot(np.log10(cs), coefs_)
ymin, ymax = pl.ylim()
pl.xlabel('log(C)')
pl.ylabel('Coefficients')
pl.title('Logistic Regression Path')
pl.axis('tight')
pl.show()
"""