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
import json
f_name = open('sochi.txt', 'r')
sochi = []
with f_name as f:
    for line in f:
        sochi.append(json.loads(line)['text'])
sochi = list(set(sochi))

f_name = open('ukraine.txt', 'r')
ukraine = []
with f_name as f:
    for line in f:
        ukraine.append(json.loads(line)['text'])
ukraine = list(set(ukraine))

text_train = sochi + ukraine
y_train = np.zeros(len(sochi) + len(ukraine))
y_train[:len(sochi)] = 1

from pprint import pprint
from time import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier()),
])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
    'vect__max_df': (0.5, 0.75, 1.0),
    #'vect__max_features': (None, 5000, 10000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    #'tfidf__use_idf': (True, False),
    #'tfidf__norm': ('l1', 'l2'),
    'clf__alpha': (0.00001, 0.000001),
    'clf__penalty': ('l2', 'elasticnet'),
    #'clf__n_iter': (10, 50, 80),
}

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

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


f_name = open('test.txt', 'r')
text_test = []
test_id = []
with f_name as f:
    for line in f:
        text_test.append(json.loads(line)['text'])
        test_id.append(json.loads(line)['id'])
y_pred = grid_search.best_estimator_.predict(text_test)

outfile = 'predictions.txt'
with open(outfile,"w+") as f:
    for i, test_id in enumerate(test_id):
        if y_pred[i] == 1:
            out_text = 'sochi'
        elif y_pred[i] == 0:
            out_text = 'ukraine'
        else:
            print "ERROR: non-binary predictions"
        f.write("%d %s\n" % (test_id, out_text))