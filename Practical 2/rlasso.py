# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 08:43:20 2014

@author: vincentli2010
"""

import warnings

import pylab as pl
import numpy as np
from scipy import linalg

from sklearn.linear_model import (RandomizedLasso, lasso_stability_path,
                                  LassoLarsCV)
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc, precision_recall_curve
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.utils.extmath import pinvh


import pickle
X = np.load(open('/home/vincentli2010/Desktop/train_main.npy', 'rb'))
_, _, y, _ = pickle.load(open('features.p', 'rb'))

with warnings.catch_warnings():
    warnings.simplefilter('ignore', UserWarning)
    lars_cv = LassoLarsCV(cv=6).fit(X, y)

# Run the RandomizedLasso: we use a paths going down to .1*alpha_max
# to avoid exploring the regime in which very noisy variables enter
# the model


alpha_grid, scores_path = lasso_stability_path(X, y, scaling=0.5, random_state=None, n_resampling=200,
                     n_grid=100, sample_fraction=0.75, eps=8.8817841970012523e-16,
                     n_jobs=-1, verbose=False)


lars_cv = LassoLarsCV(fit_intercept=True, normalize=True, precompute='auto',
                            max_iter=X.shape[1]+1000, max_n_alphas=X.shape[1]+1000,
                            eps= 2.2204460492503131e-16,copy_X=True,
                            cv=5, n_jobs=-1).fit(X,y)

alphas = np.linspace(lars_cv.alphas_[0], .1 * lars_cv.alphas_[0], 6)
clf = RandomizedLasso(alpha=lars_cv.alpha_, random_state=42, n_jobs=-1).fit(X, y)
trees = ExtraTreesRegressor(100).fit(X, y)
# Compare with F-score
F, _ = f_regression(X, y)

pl.figure()
for name, score in [('F-test', F),
                    ('Stability selection', clf.scores_),
                    ('Lasso coefs', np.abs(lars_cv.coef_)),
                    ('Trees', trees.feature_importances_),
                    ]:
    precision, recall, thresholds = precision_recall_curve(coef != 0,
                                                           score)
    pl.semilogy(np.maximum(score / np.max(score), 1e-4),
                label="%s. AUC: %.3f" % (name, auc(recall, precision)))

pl.plot(np.where(coef != 0)[0], [2e-4] * n_relevant_features, 'mo',
        label="Ground truth")
pl.xlabel("Features")
pl.ylabel("Score")
# Plot only the 100 first coefficients
pl.xlim(0, 100)
pl.legend(loc='best')
pl.title('Feature selection scores - Mutual incoherence: %.1f'
         % mi)

pl.show()