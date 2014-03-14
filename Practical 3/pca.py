# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 23:42:34 2014

@author: vincentli2010
"""


import numpy as np
import pylab as pl

from sklearn.preprocessing import StandardScaler

#X = np.load(open('x_train3', 'rb'))
X_dll = np.load(open('dll_matrix', 'rb'))
#X = np.concatenate((X, X_dll), axis=1)
X = X_dll
y = np.array(np.load(open('y_train', 'rb'))).flatten()

# Transformation
X = np.log(X + 1)
X = StandardScaler().fit_transform(X)

from sklearn.decomposition import PCA
pca = PCA(n_components=100)
pca.fit(X)
print('explained variance: %f'
      % np.sum(pca.explained_variance_ratio_))
