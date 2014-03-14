# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 15:18:29 2014

@author: vincentli2010
"""

m = np.concatenate((pred_svm[:,np.newaxis], pred_lg[:,np.newaxis], pred_rf[:,np.newaxis],pred_knn[:,np.newaxis]), axis=1)
final = np.zeros(len(pred_rf))
for i in xrange(m.shape[0]):
    final[i] = np.argmax(np.bincount(m[i,:]))
print accuracy_score(final, y_test)