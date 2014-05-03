# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 20:23:44 2014

@author: vincentli2010
"""

import numpy as np
import pickle
raw = pickle.load(open('capsule_data', 'r'))
data = np.empty((len(raw), 4))
for i, sample in enumerate(raw):
  data[i, :3] = sample[0]
  data[i, 3] = sample[1]

X_train = data[:, :3]
y_train = data[:, 3]

# LDA performance: 0.84076 +- 0.00519
# LDA with class prior: same
from sklearn import cross_validation
from sklearn.lda import LDA
classpriors = np.array([2/3.0, 1/3.0])
lda = LDA(priors=classpriors)
lda_cv_score = cross_validation.cross_val_score(lda, X_train, y_train,
                                                cv=5, n_jobs=-1)
print "LDA: %.5f +- %.5f" % (np.mean(lda_cv_score), np.std(lda_cv_score))
