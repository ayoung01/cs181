# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 20:23:44 2014

@author: vincentli2010
"""

import numpy as np
import pylab as pl

from sklearn.preprocessing import StandardScaler
from sklearn.lda import LDA

X = np.load(open('x_train3', 'rb'))
y = np.array(np.load(open('y_train', 'rb'))).flatten()

# Transformation
X = np.log(X + 1)
X = StandardScaler().fit_transform(X)

X_train, X_test = X[:2468], X[2468:]
y_train, y_test = y[:2468], y[2468:]

# LDA performance: 0.84076 +- 0.00519
# LDA with class prior: same
from sklearn import cross_validation
classpriors = np.array([3.69, 1.62, 1.2, 1.03, 1.33, 1.26, 1.72,
               1.33, 52.14, 0.68, 17.56, 1.04, 12.18,
               1.91, 1.30]) / 100
lda = LDA(priors=classpriors)
lda_cv_score = cross_validation.cross_val_score(lda, X_train, y_train,
                                                cv=5, n_jobs=2)
print "LDA: %.5f +- %.5f" % (np.mean(lda_cv_score), np.std(lda_cv_score))



model = 'LDA'
miss = np.zeros((15,15), dtype=int)
y_pred = lda.fit(X_train, y_train).predict(X_test)
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

"""
# QDA 0.79164 +- 0.02957 for {'reg_param': 0.0054286754393238594}
# Singularity issues with collinear features, so have to X_select
from sklearn.linear_model import LogisticRegression
best_estimator = LogisticRegression(C = 1.5, penalty='l1', fit_intercept=True)
best_estimator.fit(X_train, y_train)
coef_l1_LR = best_estimator.coef_.ravel()
sparsity_l1_LR = np.mean(coef_l1_LR == 0) * 100
print sparsity_l1_LR
X_selected = best_estimator.transform(X_train, 8)

from sklearn.grid_search import GridSearchCV
from sklearn.qda import QDA
cs = np.logspace(0, 3) * 1e-3
tuned_parameters = [{'reg_param': cs}]
qda = QDA(priors=None, reg_param=0.0)
qda_cv = GridSearchCV(qda, tuned_parameters,
                         cv=5, n_jobs=3, refit= False)
qda_cv.fit(X_selected, y_train)
for params, mean_score, scores in qda_cv.grid_scores_:
    if mean_score == qda_cv.best_score_:
        print ("QDA %.5f +- %.5f for %s" %
            (mean_score, scores.std(), params))




# Plot 2 PCA and LDA components
from sklearn.decomposition import PCA

X = X_train
y = y_train
target_names = np.array(['Agent', 'AutoRun',
                         'FraudLoad', 'FraudPack',
                         'Hupigon', 'Krap',
                         'Lipler', 'Magania',
                         'None', 'Poison',
                         'Swizzor', 'Tdss',
                         'VB', 'Virut', 'Zbot'])

pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

lda = LDA(n_components=2)
X_r2 = lda.fit(X, y).transform(X)

# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

pl.figure()
for c, i, target_name in zip("bry", [8, 0, 13], target_names[[8, 10, 12]]):
    pl.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, label=target_name)
pl.legend()
pl.title('PCA of Malware dataset')

pl.figure()
for c, i, target_name in zip("bry", [8, 0, 13], target_names[[8, 10, 12]]):
    pl.scatter(X_r2[y == i, 0], X_r2[y == i, 1], c=c, label=target_name)
pl.legend()
pl.title('LDA of Malware dataset')

pl.show()
"""