# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 18:42:46 2014

@author: vincentli2010

Reference:
# http://scikit-learn.org/
# Author: Olivier Grisel <olivier.grisel@ensta.org>
#         Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Mathieu Blondel <mathieu@mblondel.org>
# License: BSD 3 clause

"""
import numpy as np
y = np.array(np.load(open('y_train', 'rb'))).flatten()
X_train_base = np.load(open('x_train3', 'rb'))

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pickle as pickle
text_train = pickle.load(open('grams_list','rb'))
vectorizer = CountVectorizer(ngram_range=(1,10))
X_train_order = vectorizer.fit_transform(text_train)
X_train_order = TfidfTransformer().fit_transform(X_train_order)

#from sklearn.decomposition import RandomizedPCA
#pca = RandomizedPCA(n_components=100).fit(X_train_order)
#print pca.explained_variance_ratio_
#print np.sum(pca.explained_variance_ratio_)
#X_train_order = pca.transform(X_train_order)

#X_train = np.concatenate((X_train_base, X_train_order), axis=1)
X = X_train_order

X_train, X_test = X[:2468], X[2468:]
y_train, y_test = y[:2468], y[2468:]

from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(penalty = 'elasticnet', alpha = 0.00001)
clf.fit(X_train, y_train)
print clf.score(X_test, y_test)

"""
from sklearn.ensemble import ExtraTreesClassifier
erf = ExtraTreesClassifier(n_estimators=300, max_features='auto',
                           bootstrap=True, oob_score=True,
                           criterion='gini',
                           max_depth=None, min_samples_split=2,
                           min_samples_leaf=1,
                           n_jobs=1,
                           random_state=None, verbose=0, min_density=None,
                           compute_importances=None)
erf.fit(X_train, y_train)
print "oob\t%.4f" % (erf.oob_score_)
"""

y_pred = clf.predict(X_test)
print clf.score(X_test, y_test)
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









from pprint import pprint
from time import time

from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier


pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', ExtraTreesClassifier(n_estimators=300, max_features=20,
                           bootstrap=True, oob_score=True,
                           criterion='gini',
                           max_depth=None, min_samples_split=2,
                           min_samples_leaf=1,
                           n_jobs=-1,
                           random_state=None, verbose=0, min_density=None,
                           compute_importances=None)),
])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
    'vect__max_df': (0.5, 0.75, 1.0),
    #'vect__max_features': (None, 5000, 10000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    #'tfidf__use_idf': (True, False),
    #'tfidf__norm': ('l1', 'l2'),
    #'clf__alpha': (0.00001, 0.000001),
    #'clf__penalty': ('l2', 'elasticnet'),
    #'clf__n_iter': (10, 50, 80),
}

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, cv=5, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(text_train, y_train)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
