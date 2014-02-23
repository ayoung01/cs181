from collections import Counter
from scipy import sparse
from scipy.sparse import linalg as splinalg
import numpy as np
import util
import pickle as pickle

X_train,global_feat_dict,y_train,train_ids = \
            pickle.load(open('processed_data.p', 'rb'))
n, p = X_train.shape

print "Scatter Matrix"   
from pandas.tools.plotting import scatter_matrix
from pandas import DataFrame

avg, prop, pos, neg, posprop, negprop = \
            pickle.load(open('avg_prop_pos_neg_posprop_negprop.pickle', 'rb'))
wordcounts = pickle.load(open('wordcounts.pickle', 'rb'))
avg = np.array(avg)[:, np.newaxis]
prop = np.array(prop)[:, np.newaxis]
pos = np.array(pos)[:, np.newaxis]
neg = np.array(neg)[:, np.newaxis]
posprop = np.array(posprop)[:, np.newaxis]
negprop = np.array(negprop)[:, np.newaxis]
wordcounts = np.array(wordcounts)[:, np.newaxis]
genres = np.load(open('genres.npy', 'rb'))
ratings = np.load(open('ratings.npy', 'rb'))
companies = np.load(open('companies.npy', 'rb'))

#features = np.concatenate((avg, wordcounts), axis=1)
#features = genres[:,range(14, 20)]
#features = ratings
features = companies[:,range(5, 10)]

y_train = np.log(y_train)
mask = np.array(y_train < 14)
y_train = y_train[mask]

#features += 0.1
#features = np.log(features[mask, :])
features = features[mask, :]

df = DataFrame(np.concatenate((y_train[:, np.newaxis], features), axis=1))
scatter_matrix(df, alpha=0.2, figsize=(15, 15), diagonal='kde')