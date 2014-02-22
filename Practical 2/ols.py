## This file provides starter code for extracting features from the xml files and
## for doing some learning.
##
## The basic set-up: 
## ----------------
## main() will run code to extract features, learn, and make predictions.
## 
## extract_feats() is called by main(), and it will iterate through the 
## train/test xml files and extract each instance into a util.MovieData object.
## It will then use a series of "feature-functions" that you will write/modify
## in order to extract dictionaries of features from each util.MovieData object.
## Finally, it will produce an N x D sparse design matrix containing the union
## of the features contained in the dictionaries produced by your "feature-functions."
## This matrix can then be plugged into your learning algorithm.
##
## The learning and prediction parts of main() are largely left to you, though
## it does contain code for naive linear regression and prediction so you
## have a sense of where/what to modify.
##
## Feature-functions:
## --------------------
## "feature-functions" are functions that take a util.MovieData object representing
## a single movie, and return a dictionary mapping feature names to their respective
## numeric values. 
## For instance, a simple feature-function might map a movie object to the
## dictionary {'company-Fox Searchlight Pictures': 1}. This is a boolean feature
## indicating whether the production company of this move is Fox Searchlight Pictures,
## but of course real-valued features can also be defined. Because this feature-function
## will be run over MovieData objects for each movie instance, we will have the (different)
## feature values of this feature for each movie, and these values will make up 
## one of the columns in our final design matrix.
## Of course, multiple features can be defined within a single dictionary, and in
## the end all the dictionaries returned by feature functions will be unioned
## so we can collect all the feature values associated with a particular instance.
##
## Two example feature-functions, metadata_feats() and unigram_feats() are defined
## below. These extract metadata and unigram text features, respectively.
##
## What you need to do:
## --------------------
## 1. Write new feature-functions (or modify the example feature-functions) to
## extract useful features for this prediction task.
## 2. Implement an algorithm to learn from the design matrix produced, and to
## make predictions on unseen data. Naive code for these two steps is provided
## below, and marked by TODOs.


from collections import Counter
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg

import util

def extract_feats(ffs, datafile="train.xml", global_feat_dict=None):
    """
    arguments:
      ffs are a list of feature-functions.
      datafile is an xml file (expected to be train.xml or testcases.xml).
      global_feat_dict is a dictionary mapping feature_names to column-numbers; it
      should only be provided when extracting features from test data, so that 
      the columns of the test matrix align correctly.

    returns: 
      a sparse design matrix, a dict mapping features to column-numbers,
      a vector of target values, and a list of movie ids in order of their
      rows in the design matrix
    """
    fds = [] # list of feature dicts
    targets = []
    ids = [] 
    begin_tag = "<instance" # for finding instances in the xml file
    end_tag = "</instance>"
    in_instance = False
    curr_inst = [] # holds lines of file associated with current instance

    # start iterating thru file
    with open(datafile) as f:
        # get rid of first two lines
        _ = f.readline()
        _ = f.readline()
        for line in f:
            if begin_tag in line:
                if in_instance: 
                    assert False  # cannot have nested instances
                else:
                    curr_inst = [line]
                    in_instance = True
            elif end_tag in line:
                # we've read in an entire instance; we can extract features
                curr_inst.append(line)
                # concatenate the lines we've read and parse as an xml element
                movie_data = util.MovieData(ET.fromstring("".join(curr_inst)))
                rowfd = {}
                # union the output of all the feature functions over this instance
                
                [rowfd.update(ff(movie_data)) for ff in ffs]
                # add the final dictionary for this instance to our list
                fds.append(rowfd)
                # add target val
                targets.append(movie_data.target)
                # keep track of the movie id's for later
                ids.append(movie_data.id)
                # reset
                curr_inst = []
                in_instance = False
            elif in_instance:
                curr_inst.append(line)

    X,feat_dict = make_design_mat(fds,global_feat_dict)
    return X, feat_dict, np.array(targets), ids


def make_design_mat(fds, global_feat_dict=None):
    """
    arguments:
      fds is a list of feature dicts (one for each row).
      global_feat_dict is a dictionary mapping feature_names to column-numbers; it
      should only be provided when extracting features from test data, so that 
      the columns of the test matrix align correctly.
       
    returns: 
        a sparse NxD design matrix, where N == len(fds) and D is the number of
        the union of features defined in any of the fds 
    """
    if global_feat_dict is None:
        all_feats = set()
        [all_feats.update(fd.keys()) for fd in fds]
        feat_dict = dict([(feat, i) for i, feat in enumerate(sorted(all_feats))])
    else:
        feat_dict = global_feat_dict
        
    cols = []
    rows = []
    data = []        
    for i in xrange(len(fds)):
        temp_cols = []
        temp_data = []
        for feat,val in fds[i].iteritems():
            try:
                # update temp_cols iff update temp_data
                temp_cols.append(feat_dict[feat])
                temp_data.append(val)
            except KeyError as ex:
                if global_feat_dict is not None:
                    pass  # new feature in test data; nbd
                else:
                    raise ex

        # all fd's features in the same row
        k = len(temp_cols)
        cols.extend(temp_cols)
        data.extend(temp_data)
        rows.extend([i]*k)

    assert len(cols) == len(rows) and len(rows) == len(data)
   

    X = sparse.csr_matrix((np.array(data),
                   (np.array(rows), np.array(cols))),
                   shape=(len(fds), len(feat_dict)))
    return X, feat_dict
    

## Here are two example feature-functions. They each take in a util.MovieData
## object, and return a dictionary mapping feature-names to numeric values.
## TODO: modify these functions, and/or add new ones.

inst = 0
def feats(md):
    """
    arguments:
      md is a util.MovieData object
    returns:
      a dictionary containing a mapping from a subset of the possible metadata features
      to their values on this util.MovieData object
    """
    global inst
    inst = md
    d = {}
    
    try: 
        d['number_of_screens']  =   float(md.__dict__['number_of_screens'])  
    except:
        d['number_of_screens']  = -1
    
    try:
        d['production_budget']  =   float(md.__dict__['production_budget'])
    except KeyError:
        d['production_budget']  = -1
        
    try:
        d['running_time']       =   float(md.__dict__['running_time'])
    except KeyError:
        d['running_time']       = -1
        
    d['summer_release']         =   float(md.__dict__['summer_release'])
    d['christmas_release']      =   float(md.__dict__['christmas_release'])
    d['memorial_release']       =   float(md.__dict__['memorial_release'])
    d['independence_release']   =   float(md.__dict__['independence_release'])
    d['labor_release']          =   float(md.__dict__['labor_release'])
    
    try: 
        d['oscar_winning_directors_present'] = \
            float(md.__dict__['oscar_winning_directors_present'])
        d['num_oscar_winning_directors']     = \
            float(md.__dict__['num_oscar_winning_directors'])     
    except KeyError:
        d['oscar_winning_directors_present']    = 0.0
        d['num_oscar_winning_directors']        = 0.0
        
    
    try: 
        d['oscar_winning_actors_present'] = \
            float(md.__dict__['oscar_winning_actors_present'])
        d['num_oscar_winning_actors']     = \
            float(md.__dict__['num_oscar_winning_actors'])     
    except KeyError:
        d['oscar_winning_actors_present']    = 0.0
        d['num_oscar_winning_actors']        = 0.0
        
        
    try: 
        d['highgest_grossing_actors_present'] = \
            float(md.__dict__['highgest_grossing_actors_present'])
        d['num_highest_grossing_actors']     = \
            float(md.__dict__['num_highest_grossing_actors'])     
    except KeyError:
        d['highgest_grossing_actors_present']    = 0.0
        d['num_highest_grossing_actors']        = 0.0
        
    return d
    

"""
## The following function does the feature extraction, learning, and prediction
trainfile = "train.xml"
testfile = "testcases.xml"
outputfile = "mypredictions.csv"  # feel free to change this or take it as an argument

# TODO put the names of the feature functions you've defined above in this list
ffs = [feats]

# extract features
print "extracting training features..."
X_train,global_feat_dict,y_train,train_ids = extract_feats(ffs, trainfile)
print "done extracting training features"
X_train = X_train.toarray()
n, p = X_train.shape


import pickle as pickle
pickle.dump((X_train, global_feat_dict, y_train, train_ids),
            open('processed_data.p', 'wb'))
"""
import pickle as pickle
X_train,global_feat_dict,y_train,train_ids = \
            pickle.load(open('processed_data.p', 'rb'))
n, p = X_train.shape

# missing value imputation, missing value is encoded as -1    
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values= -1, strategy='mean', axis=0)
X_train = imp.fit_transform(X_train)


print "Scatter Matrix"   
from pandas.tools.plotting import scatter_matrix
from pandas import DataFrame


X_train_origin = X_train.copy()
y_train_origin = y_train.copy()

# number_of_screens should be raised to the 8th power
# production_budget is mostly linear, somewhat 2nd degree
y_train = y_train_origin.copy()
X_train = X_train_origin.copy()


y_train = np.log(y_train)
X_train[:, global_feat_dict['number_of_screens']] = \
    np.log(X_train[:, global_feat_dict['number_of_screens']]) ** 11
X_train[:, global_feat_dict['production_budget']] = \
    np.log(X_train[:, global_feat_dict['production_budget']]) ** 3



quant_set = [global_feat_dict['running_time'], global_feat_dict['number_of_screens'], \
                global_feat_dict['production_budget']]
df = DataFrame(np.concatenate((y_train[:, np.newaxis], X_train[:,quant_set]), axis=1))
scatter_matrix(df, alpha=0.2, figsize=(p+1, p+1), diagonal='kde')


"""
# TODO train here, and return regression parameters
print "learning..."
learned_w = splinalg.lsqr(X_train,y_train)[0]
print "done learning"
print

# get rid of training data and load test data
del X_train
del y_train
del train_ids
print "extracting test features..."
X_test,_,y_ignore,test_ids = extract_feats(ffs, testfile, global_feat_dict=global_feat_dict)
print "done extracting test features"
print

# TODO make predictions on text data and write them out
print "making predictions..."
preds = X_test.dot(learned_w)
print "done making predictions"
print

print "writing predictions..."
util.write_predictions(preds, test_ids, outputfile)
print "done!"
"""