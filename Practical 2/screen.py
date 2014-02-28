# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 14:47:08 2014

@author: vincentli2010

"""

import numpy as np
import pickle
discard, global_feat_dict, y_train, train_ids = pickle.load(open('features.p', 'rb'))
grams = np.load(open('/home/vincentli2010/Desktop/allgrams.npy', 'rb'))

BETA = np.empty((grams.shape[1],))
from sklearn.linear_model import LinearRegression
ols = LinearRegression(fit_intercept=True,normalize=False,copy_X=True)

for j in range(grams.shape[1]):
    x = grams[:,j][:,np.newaxis]
    y = y_train
    ols.fit(x, y)
    BETA[j] = ols.coef_[0] / np.sum(x != 0)
"""
mask = np.abs(BETA) > (BETA.mean() + 4 * BETA.std())
gram_4std = grams[:,mask]; print gram_4std.shape #226
np.save(open('/home/vincentli2010/Desktop/gram_4std.npy', 'wb'), gram_4std)

mask = np.abs(BETA) > (BETA.mean() + 3 * BETA.std())
gram_3std = grams[:,mask]; print gram_3std.shape #559
np.save(open('/home/vincentli2010/Desktop/gram_3std.npy', 'wb'), gram_3std)

mask = np.abs(BETA) > (BETA.mean() + 2 * BETA.std())
gram_2std = grams[:,mask]; print gram_2std.shape #1422
np.save(open('/home/vincentli2010/Desktop/gram_2std.npy', 'wb'), gram_2std)

mask = np.abs(BETA) > (BETA.mean() + 1 * BETA.std())
gram_1std = grams[:,mask]; print gram_1std.shape #4141
np.save(open('/home/vincentli2010/Desktop/gram_1std.npy', 'wb'), gram_1std)

mask = np.abs(BETA) > (BETA.mean() + 0 * BETA.std())
gram_0std = grams[:,mask]; print gram_0std.shape #17888
np.save(open('/home/vincentli2010/Desktop/gram_0std.npy', 'wb'), gram_0std)
"""

mask = np.abs(BETA) > (BETA.mean() + 4 * BETA.std())
gram_4std = grams[:,mask]; print gram_4std.shape #226
np.save(open('/home/vincentli2010/Desktop/gram_scale_4std.npy', 'wb'), gram_4std)

mask = np.abs(BETA) > (BETA.mean() + 3 * BETA.std())
gram_3std = grams[:,mask]; print gram_3std.shape #475
np.save(open('/home/vincentli2010/Desktop/gram_scale_3std.npy', 'wb'), gram_3std)

mask = np.abs(BETA) > (BETA.mean() + 2 * BETA.std())
gram_2std = grams[:,mask]; print gram_2std.shape #1080
np.save(open('/home/vincentli2010/Desktop/gram_scale_2std.npy', 'wb'), gram_2std)

mask = np.abs(BETA) > (BETA.mean() + 1 * BETA.std())
gram_1std = grams[:,mask]; print gram_1std.shape #2820
np.save(open('/home/vincentli2010/Desktop/gram_scale_1std.npy', 'wb'), gram_1std)

mask = np.abs(BETA) > (BETA.mean() + 0 * BETA.std())
gram_0std = grams[:,mask]; print gram_0std.shape #16143
np.save(open('/home/vincentli2010/Desktop/gram_scale_0std.npy', 'wb'), gram_0std)
