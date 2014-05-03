# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 20:23:44 2014

@author: vincentli2010
"""

import numpy as np


import pickle
raw = pickle.load(open('capsule_data_new', 'r'))
data = np.empty((len(raw), 4))
for i, sample in enumerate(raw):
  data[i, :3] = sample[0]
  data[i, 3] = sample[1]

X_train = data[:, :3]
y_train = data[:, 3]


#raw = np.genfromtxt('/home/vincentli2010/Desktop/ghost_train.csv', delimiter=',')[:10000]
#np.save(open('ghost_train.npy', 'wb'), raw)
"""
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
"""
# LDA performance: 0.84076 +- 0.00519
# LDA with class prior: same
from sklearn import cross_validation
from sklearn.lda import LDA
#classpriors = np.array([2/3.0, 1/3.0])
lda = LDA(priors=None)
lda_cv_score = cross_validation.cross_val_score(lda, X_train, y_train,
                                                cv=5, n_jobs=-1)
print "LDA: %.5f +- %.5f" % (np.mean(lda_cv_score), np.std(lda_cv_score))
lda.fit(X_train, y_train)

from sklearn.externals import joblib
joblib.dump(lda, 'capsule_predictor_lda.pkl', compress=9)
#ghost_predictor = joblib.load('ghost_predictor.pkl')
