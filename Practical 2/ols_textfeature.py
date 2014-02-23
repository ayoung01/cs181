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
    try:
        d['num_highest_grossing_actors'] = md.__dict__['num_highest_grossing_actors']
    except:
        d['num_highest_grossing_actors'] = 0
    return d
    
    
## The following function does the feature extraction, learning, and prediction
trainfile = "train.xml"
testfile = "testcases.xml"
outputfile = "mypredictions.csv"  # feel free to change this or take it as an argument

# TODO put the names of the feature functions you've defined above in this list
ffs = [feats]

# extract features
print "extracting training features..."
X_train,y_train,train_ids = extract_feats(ffs, trainfile)
print "done extracting training features"
genres_uniq = []; origins_uniq = []; genres = []; release_dates = [];
origins = []; ratings = []; companies = []; hi_actors = [];
for features in X_train:
    genres.append(features['genres'])
    genres_uniq.extend(features['genres'])
    release_dates.append(features['release_date'])
    origins_uniq.extend(features['origins'])
    origins.append(features['origins'])
    ratings.append(features['rating'])
    companies.append(features['company'])
    hi_actors.append(features['num_highest_grossing_actors'])
    
genres_uniq = list(set(genres_uniq))
origins_uniq = list(set(origins_uniq))
ratings_uniq = list(set(ratings))
companies_uniq = list(set(companies))

"""
0  1 SuspenseThriller 2 Noir 3 Crime 4 Romance 5 Animation 6 Scifi 7 Comedy
8 War 9 Horror 10 Western 11 Adventure 12 Mystery 13 Short 14 Foreign 15 Drama
16 Action 17 Documentary 18 Musical 19 FamilyKids 20 Fantasy 21 GayLesbian

Counts per genre:
1 SuspenseThriller 223 Noir 1 Crime 94 Romance 152 Animation 39 Scifi 46
Comedy 387 War 28 Horror 90 Western 6 Adventure 114 Mystery 48 Short 1
Foreign 211 Drama 651 Action 147 Documentary 137 Musical 48 FamilyKids 72 Fantasy
75 GayLesbian 18
"""
genres_ind = []; companies_ind = [];

usa = []; foreign = [];

# indicator arrays for ratings
g = np.array([int(x=='G') for x in ratings])[:, np.newaxis]
pg = np.array([int(x=='PG') for x in ratings])[:, np.newaxis]
pg13 = np.array([int(x=='PG13' or x=='PG-13') for x in ratings])[:, np.newaxis]
r = np.array([int(x=='R') for x in ratings])[:, np.newaxis]
nc17 = np.array([int(x=='NC-17') for x in ratings])[:, np.newaxis]

ratingnp = np.concatenate((g, pg, pg13, r, nc17), axis = 1)
np.save(open('ratings.npy', 'wb'), ratingnp)

for genre in genres_uniq:
    genres_ind.append([int(genre in x) for x in genres])
np.save(open('genres.npy', 'wb'), np.array(genres_ind).T)

for company in companies_uniq:
    ind = [int(x==company) for x in companies]
    if sum(ind) > 20:
        companies_ind.append(ind)
np.save(open('companies.npy', 'wb'), np.array(companies_ind).T)

np.save(open('hi_actors.npy', 'wb'), np.array(hi_actors).T)
#companies_by_freq = []
#for i, company in enumerate(companies_uniq):
#    companies_by_freq.append((sum(companies_ind[i]), company))
#print sorted(companies_by_freq, reverse = True)
"""
[(44, 'Universal Pictures'), (41, 'Sony Pictures Classics'), (40, 'IFC Films'),
(31, 'New Line Cinema'), (30, 'Paramount Pictures'), (29, 'Buena Vista Pictures'),
(28, 'Miramax Films'), (26, 'Warner Bros. Pictures'), (26, 'Magnolia Pictures'),
(25, 'Warner Bros.'), (25, 'Fox Searchlight Pictures'),
(23, 'Columbia Pictures / Sony Pictures Entertainment'),
(21, 'Strand Releasing'), (20, 'ThinkFilm'), (19, 'The Weinstein Company'),
(19, 'Lions Gate Films Inc.'), (19, 'Lions Gate Films'), (19, 'Focus Features'),
(16, 'Warner Independent Pictures'), (16, 'Twentieth Century Fox Film Corporation'),
(16, 'Palm Pictures'), (14, 'Twentieth Century-Fox Film Corporation'), (14, 'Regent Releasing'),
(14, 'Metro-Goldwyn-Mayer'), (12, 'Twentieth Century Fox Film Corp.'), (12, 'IFC First Take'),
(11, 'Wellspring Media'), (11, 'Tartan USA'), (10, 'ThinkFilm Inc.'), (10, 'Roadside Attractions'),
(10, 'Paramount Vantage'), (10, 'First Run Features'), (9, 'TLA Releasing'), (9, 'THINKFilm'),
(9, 'New Yorker Films'), (8, 'Paramount Classics'), (8, 'MGM'), (8, 'First Independent Pictures'), (8, 'Columbia Pictures / Sony Pictures Releasing'), (7, 'Sony Pictures Releasing'), (7, 'Sony Pictures Entertainment'), (7, 'Samuel Goldwyn Films'), (7, 'Picturehouse'), (7, 'Lionsgate'), (7, 'Kino International'), (7, 'First Look Pictures Releasing'), (7, 'DreamWorks SKG'), (7, 'Dimension Films'), (6, 'Zeitgeist Films'), (6, 'Yari Film Group Releasing'), (6, 'Walt Disney Pictures'), (6, 'Touchstone Pictures'), (6, 'Rogue Pictures'), (6, 'Newmarket Films'), (6, 'Indican Pictures'), (6, 'DreamWorks Distribution LLC'), (6, 'Columbia Pictures'), (5, 'TriStar Pictures / Sony Pictures Entertainment'), (5, 'Sony Pictures'), (5, 'Screen Gems Inc.'), (5, 'Picture This! Entertainment'), (5, 'Koch Lorber Films'), (5, 'Kino International Corp.'), (5, 'Freestyle Releasing LLC'), (5, 'Freestyle Releasing'), (5, 'Cinema Libre Studio'), (5, '20th Century Fox'), (4, 'Yari Film Group'), (4, 'Samuel Goldwyn Films LLC'), (4, 'Cinema Guild'), (4, 'Balcony Releasing'), (3, 'Truly Indie'), (3, 'The Weinstein Company LLC'), (3, 'Seventh Art Releasing'), (3, 'Screen Media Films'), (3, 'Screen Gems'), (3, 'Picturehouse Entertainment'), (3, 'Lifesize Entertainment'), (3, 'First Look Pictures'), (3, 'Emerging Pictures'), (3, 'Bauer Martinez Studios'), (3, 'Anchor Bay Entertainment'), (3, '20th Century Fox Pictures'), (2, 'Warner Bros. Pictures Inc.'), (2, 'Universal Studios Inc.'), (2, 'Thinkfilm'), (2, 'The 7th Floor'), (2, 'Summit Entertainment'), (2, 'Slowhand Cinema Releasing'), (2, 'Rainbow Releasing'), (2, 'Paramount Pictures / DreamWorks SKG'), (2, 'Metro-Goldwyn-Mayer Distributing Corporation'), (2, 'Leisure Time Features'), (2, 'International Film Circuit'), (2, 'Goodbye Cruel Releasing'), (2, 'Fox Searchlight'), (2, 'First Look Media'), (2, 'First Look International'), (2, 'Film Movement'), (2, 'Empire Pictures'), (2, 'DreamWorks Pictures'), (2, 'Codeblack Entertainment'), (2, 'City Lights Pictures'), (2, 'Cineville Inc.'), (2, 'Castle Hill Productions'), (2, 'Buena Vista Pictures / Walt Disney Pictures'), (2, 'Artistic License'), (2, 'ArtMattan Productions'), (2, 'Ariztical Entertainment Inc.'), (2, 'After Dark Films'), (2, True), (1, 'Youth House Productions / Fader Films'), (1, 'Wolfe Releasing'), (1, 'Walt Disney Pictures / IMAX Corporation'), (1, 'Vitagraph Films'), (1, 'Urban Vision Entertainment'), (1, 'Universal Studios'), (1, 'Universal Pictures / HBO Films'), (1, 'Universal Pictures / DreamWorks Distribution LLC'), (1, 'Universal Picture'), (1, 'United Artists (MGM)'), (1, 'Typecast Releasing'), (1, 'Toho Company Ltd.'), (1, 'The Weinstein Company LLC / IFC Films'), (1, 'The Weinstein Company / MGM'), (1, 'The Weinstein Company /  Metro-Goldwyn-Mayer'), (1, 'The Samuel Goldwyn Company'), (1, 'The Disinformation Company'), (1, 'Stonehaven Media'), (1, 'Stick Figure Productions'), (1, 'Stardust Pictures / Cinema Libre Studio'), (1, 'Sony Pictures Studios'), (1, 'Sony Pictures Entertainment Inc.'), (1, 'Sony Pictures Classics / United Artists'), (1, 'Sony BMG'), (1, 'Slowhand Releasing'), (1, 'Slowhand Cinema'), (1, 'Slow Hand Releasing'), (1, 'Silver Nitrate Releasing'), (1, 'Shout! Factory'), (1, 'ShadowCatcher Entertainment'), (1, 'Shadow Distribution Inc.'), (1, 'Shadow Distribution'), (1, 'SenArt Films'), (1, 'Screen Media Films LLC'), (1, 'Screen Gems Inc. / Sony Pictures Entertainment'), (1, 'Screen Gems (Sony)'), (1, 'Samuel Goldwyn Films LLC / Roadside Attractions'), (1, 'Samuel Goldwyn Films LLC  / Roadside Attractions'), (1, 'Samuel Goldwyn Films / Sony Pictures Entertainment'), (1, 'Samuel Goldwyn Films / Roadside Attractions'), (1, 'Rumur Releasing'), (1, 'Romar Entertainment'), (1, 'Roadside Attractions / Samuel Goldwyn Films LLC'), (1, 'Roadside Attractions / Samuel Goldwyn Company'), (1, 'Revolution Studios'), (1, 'Regent Entertainment'), (1, 'Red Envelope Entertainment / Truly Indie'), (1, 'Quiet Man Productions'), (1, 'Promenade Pictures'), (1, 'Priority Films'), (1, 'Polychrome Pictures LLC'), (1, 'Picturehouse Entertainment LLC'), (1, 'Peace Arch Releasing'), (1, 'Pasidg Productions Inc.'), (1, 'Palm Pictures / ImaginAsian Pictures'), (1, 'Palisades Pictures'), (1, 'Outsider Pictures'), (1, 'Nu-Lite Entertainment'), (1, 'Northern Arts Entertainment'), (1, 'Newport Films'), (1, 'New Dog Distribution'), (1, 'Moonstone Entertainment'), (1, 'Monterey Media'), (1, 'Minority Films LLC'), (1, 'Millennium Films Inc.'), (1, 'Millennium Films / Samuel Goldwyn Films LLC / Roadside Attractions'), (1, 'Microangelo Entertainment LLC'), (1, 'Metro-Goldwyn-Mayer Distributing Corporation / The Weinstein Company'), (1, 'Metro-Goldwyn-Mayer / The Weinstein Company'), (1, 'Menemsha Films'), (1, 'Meadowbrook Pictures'), (1, 'Maya Entertainment'), (1, 'Matson Films'), (1, 'Marshall Curry Productions LLC'), (1, 'Margin Films'), (1, 'MTV Films / Paramount Classics'), (1, 'MGM, The Weinstein Company'), (1, 'MGM Studios'), (1, 'MGM / UA Distribution Company'), (1, 'MGM / The Weinstein Company'), (1, 'Luminous Velocity Releasing'), (1, 'Lovett Productions'), (1, 'Lionsgate / AfterDark Films'), (1, 'Lions Gate Films Inc. / City Lights Media Group'), (1, 'Lions Gate Films / DEJ Productions'), (1, 'Laemmle/Zeller Films'), (1, 'Kindred Media Group'), (1, 'Jungle Films LLC'), (1, 'ITVS'), (1, 'IMAX Corporation'), (1, 'IFC First Take Films'), (1, 'IFC Films / The Weinstein Company'), (1, 'IFC Films / Renaissance Films'), (1, 'IFC Films / Red Envelope Entertainment'), (1, 'Horizon Entertainment'), (1, 'Hollywood Pictures'), (1, 'Holedigger Studios'), (1, 'HBO/Cinemax Documentary / ThinkFilm'), (1, 'Gorilla Factory Productions'), (1, 'Gold Circle Films'), (1, 'Gigantic Pictures'), (1, 'Gaumont'), (1, 'Gabbert/Libresco Productions'), (1, 'Freestyle Releasing LLC / Yari Film Group Releasing'), (1, 'Freestlyle Releasing'), (1, 'Fox Faith / The Bigger Picture'), (1, 'Fox Atomic / 20th Century Fox'), (1, 'Fox Atomic'), (1, 'Fox 2000 Pictures'), (1, 'Forward Entertainment LLC'), (1, 'Focus Features / Universal Studios'), (1, 'Fine Line Featues'), (1, 'Fabrication Films'), (1, 'Embrem Entertainment'), (1, 'Economic Projections'), (1, 'Echo Bridge Entertainment LLC'), (1, 'DreamWorks Distribution'), (1, 'DreamWorks Animation'), (1, 'Dream Entertainment'), (1, 'Dog Lover&#039;s Symphony, LLC'), (1, 'Docurama / Shadow Distribution Inc.'), (1, 'Distribution Machine Associates'), (1, 'Disney'), (1, 'Dinsdale Releasing'), (1, 'Dimension Films / The Weinstein Company'), (1, 'Dimension Films / Metro-Goldwyn-Mayer'), (1, 'Destination Films'), (1, 'Daddy W Productions'), (1, 'Cyan Pictures'), (1, 'Crusader Entertainment LLC'), (1, 'Coquette Productions / NaVinci Films'), (1, 'Concord Media Group'), (1, 'Columbia Pictures (Sony)'), (1, 'Code Black Entertainment'), (1, 'Cockeyed Caravan'), (1, 'Celluloid Dreams / International Film Circuit'), (1, 'Celluloid Dreams'), (1, 'Celestine Films Holding Company'), (1, 'Castle Rock Entertainment / New Line Cinema'), (1, 'Buena Vista Pictures / Walt Disney Studios'), (1, 'Black Diamond Pictures'), (1, 'Baraka Productions'), (1, 'Badland Corporation'), (1, 'B.D. Fox Marketing and Distribution'), (1, 'Avatar / Home Vision Entertainment'), (1, 'Autonomous Films'), (1, 'Argot Pictures'), (1, 'Americano Productions'), (1, 'AfterDark Films / Freestyle Releasing LLC'), (1, '8X Entertainment Inc.'), (1, '518 Media, Inc.'), (1, '20th Century Fox / Fox Faith')]

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