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
import pickle
import re
from dateutil import parser

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

    #X,feat_dict = make_design_mat(fds,global_feat_dict)
    return fds, np.array(targets), ids


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
    """
      AC - Austin Chronicle review text (string)
      BO - Boston Globe review text (string)
      CL - LA Times review text (string)
      EW - Entertainment Weekly review text (string)
      NY - New York Times review text (string)
      VA - Variety review text (string)
      VV - Village Voice review text (string)
    """
    #try:
    #    d['AC']  =   md.__dict__['AC']
    #except:
    #    d['AC']  = -1
    #try:
    #    d['BO']  =   md.__dict__['BO']
    #except:
    #    d['BO']  = -1
    #try:
    #    d['CL']  =   md.__dict__['CL']
    #except:
    #    d['CL']  = -1
    #try:
    #    d['EW']  =   md.__dict__['EW']
    #except:
    #    d['EW']  = -1
    #try:
    #    d['NY']  =   md.__dict__['NY']
    #except:
    #    d['NY']  = -1
    #try:
    #    d['VA']  =   md.__dict__['VA']
    #except:
    #    d['VA']  = -1
    #try:
    #    d['VV']  =   md.__dict__['VV']
    #except:
    #    d['VV']  = -1

    # subtract to get baseline of zero
    d['release_date'] = parser.parse(md.__dict__['release_date']).toordinal() - 731953
    d['genres']  =   md.__dict__['genres']
    d['origins']  =   md.__dict__['origins']
    d['rating'] = md.__dict__['rating']
    d['company'] = md.__dict__['company']
    d['director'] = md.__dict__['directors']
    d['actors'] = md.__dict__['actors']
    d['authors'] = md.__dict__['authors']

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
        d['highest_grossing_actors_present'] = \
            float(md.__dict__['highest_grossing_actors_present'])
        d['num_highest_grossing_actors'] = md.__dict__['num_highest_grossing_actors']
        d['highest_grossing_actor'] = md.__dict__['highest_grossing_actor']
    except:
        d['highest_grossing_actors_present']    = 0.0
        d['num_highest_grossing_actors'] = 0
        d['highest_grossing_actor'] = []

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

    return d
# We need to first get the highest grossing actors present; discard number of oscar-winning directors (always 1 or 0)
# production budget has missing values

## The following function does the feature extraction, learning, and prediction
trainfile = "train.xml"
testfile = "testcases.xml"
outputfile = "mypredictions.csv"  # feel free to change this or take it as an argument

ffs = [feats]

# extract features
print "extracting training features..."
#X_train,y_train,train_ids = extract_feats(ffs, trainfile)
X_train,y_train,train_ids = extract_feats(ffs, trainfile)


print "done extracting training features"
genres_uniq=[];origins_uniq=[];hi_actors_uniq=[];directors_uniq=[];
actors_uniq=[];authors_uniq=[];
genres=[];release_dates=[];hi_actors=[];num_hi_actors=[];origins=[];ratings=[];
companies=[];directors=[];actors=[];authors=[];num_oscar_winning_actors=[];
oscar_winning_actors_present=[];oscar_winning_directors_present=[];
num_oscar_winning_directors=[];summer=[];christmas=[];memorial=[];
independence=[];labor=[];num_screens=[];production_budget=[];running_time=[];
hi_actors_present=[];

for features in X_train:
    genres.append(features['genres'])
    genres_uniq.extend(features['genres'])
    release_dates.append(features['release_date'])
    origins_uniq.extend(features['origins'])
    origins.append(features['origins'][0])
    ratings.append(features['rating'])
    companies.append(features['company'])
    num_hi_actors.append(features['num_highest_grossing_actors'])
    hi_actors.append(features['highest_grossing_actor'])
    hi_actors_uniq.extend(features['highest_grossing_actor'])
    directors.append(features['director'])
    directors_uniq.extend(features['director'])
    actors.append(features['actors'])
    try:
        actors_uniq.extend(features['actors'])
    except:
        pass
    authors.append(features['authors'])
    authors_uniq.extend(features['authors'])
    num_oscar_winning_actors.append(features['num_oscar_winning_actors'])
    oscar_winning_actors_present.append(features['oscar_winning_actors_present'])
    oscar_winning_directors_present.append(features['oscar_winning_directors_present'])
    num_oscar_winning_directors.append(features['num_oscar_winning_directors'])
    summer.append(features['summer_release'])
    christmas.append(features['christmas_release'])
    memorial.append(features['memorial_release'])
    independence.append(features['independence_release'])
    labor.append(features['labor_release'])
    num_screens.append(features['number_of_screens'])
    production_budget.append(features['production_budget'])
    running_time.append(features['running_time'])
    hi_actors_present.append(features['highest_grossing_actors_present'])

genres_uniq = list(set(genres_uniq)); genres_uniq.pop(0) # gets rid of empty genre
origins_uniq = list(set(origins_uniq))
ratings_uniq = list(set(ratings))
companies_uniq = list(set(companies))
hi_actors_uniq = list(set(hi_actors_uniq))
directors_uniq = list(set(directors_uniq))
actors_uniq = list(set(actors_uniq))
authors_uniq = list(set(authors_uniq))

genres_ind=[];companies_ind=[];hi_actors_ind=[];origins_ind=[];
directors_ind=[];actors_ind=[];authors_ind=[];

# indicator arrays for ratings
g = np.array([int(x=='G') for x in ratings])[:, np.newaxis]
pg = np.array([int(x=='PG') for x in ratings])[:, np.newaxis]
pg13 = np.array([int(x=='PG13' or x=='PG-13') for x in ratings])[:, np.newaxis]
r = np.array([int(x=='R') for x in ratings])[:, np.newaxis]
nc17 = np.array([int(x=='NC-17') for x in ratings])[:, np.newaxis]

for hi_actor in hi_actors_uniq:
    hi_actors_ind.append([int(hi_actor in x) for x in hi_actors])

for genre in genres_uniq:
    genres_ind.append([int(genre in x) for x in genres])

# sort companies by number of movies in training set
for company in companies_uniq:
    ind = [int(x==company) for x in companies]
    companies_ind.append(ind)
companies_by_freq = []
for i, company in enumerate(companies_uniq):
    companies_by_freq.append((sum(companies_ind[i]), company))
    companies_by_freq = sorted(companies_by_freq, reverse = True)
companies_uniq_sorted = []
# truncate by top 40 companies
for x, company in companies_by_freq[:40]:
    companies_uniq_sorted.append(company)
companies_ind=[]
for company in companies_uniq_sorted:
    companies_ind.append([int(x==company) for x in companies])

# sort directors by number of movies in training set
for director in directors_uniq:
    ind = [int(director in x) for x in directors]
    directors_ind.append(ind)
directors_by_freq = []
for i, director in enumerate(directors_uniq):
    directors_by_freq.append((sum(directors_ind[i]), director))
    directors_by_freq = sorted(directors_by_freq, reverse = True)
directors_uniq_sorted = []
# truncate by top 10 directors
for x, director in directors_by_freq[:10]:
    directors_uniq_sorted.append(director)
directors_ind=[]
for director in directors_uniq_sorted:
    directors_ind.append([int(director in x) for x in directors])

# text sentiment features
avg, prop, pos, neg, posprop, negprop = \
            pickle.load(open('avg_prop_pos_neg_posprop_negprop.pickle', 'rb'))
avg = np.array(avg)[:, np.newaxis]
prop = np.array(prop)[:, np.newaxis]
pos = np.array(pos)[:, np.newaxis]
neg = np.array(neg)[:, np.newaxis]
posprop = np.array(posprop)[:, np.newaxis]
negprop = np.array(negprop)[:, np.newaxis]
X_sentiment = np.concatenate((avg, prop, pos, neg, posprop, negprop), axis=1)

sentiment_indices = ['avg', 'prop','pos', 'neg','posprop','negprop']

# review word count
X_wc = pickle.load(open('wordcounts.pickle', 'rb'))
X_wc = np.array(X_wc)[:, np.newaxis]


"""
# parse origins (cleaned manually)
origins_uniq_clean = []
for origin in origins_uniq:
    countries = re.findall('[A-Z][^A-Z]*', origin)
    origins_uniq_clean.extend(countries)
origins_uniq_clean = list(set(origins_uniq_clean))
"""
origins_uniq_clean = ['USA', 'UK', 'Canada', 'Brazil', 'HongKong', 'Italy', 'Ireland','SouthAfrica','Georgia','Thailand','France', 'Switzerland', 'Rico', 'Zealand','Norway', 'Argentina','Israel', 'Australia','Singapore', 'Iceland', 'Kazakhstan', 'China','Belgium', 'Germany','Spain','Netherlands', 'Bosnia', 'Denmark', 'Turkey', 'Finland','Korea', 'Luxembourg', 'Hungary', 'Croatia', 'Iran', 'Russia','Romania','Mexico','India', 'Sweden', 'Czech', 'Austria','Japan']

for origin in origins_uniq_clean:
    origins_ind.append([int(origin in x) for x in origins])

#for i, country in enumerate(origins_ind):
#    print (origins_uniq_clean[i], sum(origins_ind[i])),

# remove bizarre occurences of booleans in actor list
for i, actor_list in enumerate(actors):
    if True == actor_list:
        actors[i] = []
# sort actors by number of movies in training set
for actor in actors_uniq:
    ind = [int(actor in x) for x in actors]
    actors_ind.append(ind)
actors_by_freq = []
for i, actor in enumerate(actors_uniq):
    actors_by_freq.append((sum(actors_ind[i]), actor))
    actors_by_freq = sorted(actors_by_freq, reverse = True)
actors_by_freq.pop(0) # Remove empty string from actor list
actors_uniq_sorted = []
# truncate by top 85 actors (>= 6 occurrences)
for x, actor in actors_by_freq[:97]:
    if not actor.lower() in hi_actors_uniq: # avoid re-counting (removes 12 actors)
        actors_uniq_sorted.append(actor)
actors_ind=[]
for actor in actors_uniq_sorted:
    actors_ind.append([int(actor in x) for x in actors])
num_actors = [len(actor_list) for actor_list in actors]

# sort authors by number of movies in training set
for author in authors_uniq:
    ind = [int(author in x) for x in authors]
    authors_ind.append(ind)
authors_by_freq = []
for i, author in enumerate(authors_uniq):
    authors_by_freq.append((sum(authors_ind[i]), author))
    authors_by_freq = sorted(authors_by_freq, reverse = True)
authors_by_freq.pop(0) # Remove empty string from author list
authors_uniq_sorted = []
# truncate by top 29 authors (>=3 occurrences)
# 174 authors have >=2 occurrences
for x, author in authors_by_freq[:29]:
    authors_uniq_sorted.append(author)
authors_ind=[]
for author in authors_uniq_sorted:
    authors_ind.append([int(author in x) for x in authors])
num_authors = [len(author_list) for author_list in authors]

"""
8 number_of_screens
11 production_budget

14 review word count

55-60 sentiments 0.28247*** 58 -> 0.16029 # number of negative [0] sentences in review

12 running_time

5 num_highest_grossing_actors
6 num_oscar_winning_actors
7 num_oscar_winning_directors

0 christmas_release
2 independence_release
3 labor_release
4 memorial_release
13 summer_release

1 highest_grossing_actors_present
9 oscar_winning_actors_present
10 oscar_winning_directors_present

61 np_release_dates
15-36 genres
37-49 companies
50-54 ratings
"""

np_feat = np.concatenate((
            np.array(christmas)[:, np.newaxis], #0
            np.array(hi_actors_present)[:, np.newaxis], #1
            np.array(independence)[:, np.newaxis], #2
            np.array(labor)[:, np.newaxis], #3
            np.array(memorial)[:, np.newaxis], #4
            np.array(num_hi_actors)[:, np.newaxis], #5
            np.array(num_oscar_winning_actors)[:, np.newaxis], #6
            np.array(oscar_winning_directors_present)[:, np.newaxis], #7
            np.array(num_screens)[:, np.newaxis], #8
            np.array(oscar_winning_actors_present)[:, np.newaxis], #9
            np.array(num_oscar_winning_directors)[:, np.newaxis], #10
            np.array(production_budget)[:, np.newaxis], #11
            np.array(running_time)[:, np.newaxis], #12
            np.array(summer)[:, np.newaxis], #13
            X_wc, #14
            np.array(genres_ind).T, #15-36
            np.array(companies_ind).T, #37-49
            g, pg, pg13, r, nc17, #50-54
            X_sentiment, #55-60
            np.array(release_dates)[:, np.newaxis], #61
            np.array(hi_actors_ind).T,
            np.array(origins_ind).T,
            np.array(directors_ind).T,
            np.array(actors_ind).T,
            np.array(authors_ind).T,
            np.array(num_actors)[:, np.newaxis],
            np.array(num_authors)[:, np.newaxis],
            ), axis=1)

feat_indices = ['Christmas', 'Highest grossing actors present']\
    +['Independence','Labor','Memorial','Number of highest grossing actors']\
    +['Number of Oscar winning actors','Oscar winning directors present']\
    +['Number of screens','Oscar winning actors present','Number of Oscar winning directors','Production budget']\
    +['Running time','Summer','Average review word count']\
    +genres_uniq+companies_uniq_sorted+['G','PG','PG-13','R','NC-17']\
    +sentiment_indices+['Release date']\
    +hi_actors_uniq+origins_uniq_clean+directors_uniq_sorted+actors_uniq_sorted\
    +authors_uniq_sorted+['Number of actors','Number of authors']

np.save(open('feat_names.npy','wb'),np.array(feat_indices))

print "Dimensions of feature matrix: " + str(np_feat.shape)
np.save(open('feat.npy', 'wb'), np_feat)

"""
# create inputs for Stanford NLP
filelist = open('reviewlist.txt', 'w')
for i, reviews in enumerate(X_train):
    for reviewer, text in reviews.iteritems():
        if text == -1:
            continue
        else:
            filename = 'reviews/' + train_ids[i] + reviewer + '.txt'
            f = open(filename, 'w')
            f.write(text)
            f.close()
            filelist.write(filename+'\n')

# dump average review word counts for each movie
wordcounts = []
for i, reviews in enumerate(X_train):
    num_reviews = 0.0
    num_words = 0.0
    for reviewer, text in reviews.iteritems():
        if text == -1:
            continue
        else:
            wordcount = len(text.split())
            num_words += wordcount
            num_reviews += 1
    wordcounts.append(num_words/num_reviews)
pickle.dump(wordcounts, open('wordcounts.pickle', 'wb'))
"""