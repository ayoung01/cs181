import re
import os
import pickle

# {movieid_1: {review1: [1,1,0,2, ...], review2: [3,2,,3, ...], ...}, movieid_2: {...}, ...}
sentiments = {}

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

for f in os.listdir("nlp/"):
    print f
    movie_id = f[:-10]
    if not movie_id in sentiments:
        sentiments[movie_id] = {}
    review_src = f[-10:-8]
    sentiments[movie_id][review_src] = []
    tree = ET.parse("nlp/" + f)
    root = tree.getroot()
    for sentence in root.iter('sentence'):
        sentimentValue = sentence.attrib['sentimentValue']
        sentiments[movie_id][review_src].append(sentimentValue)
pickle.dump(sentiments, open('sentiments.pickle', 'wb'))

### FEATURES ###    
avg = [] # average sentiment across reviews
prop = [] # ratio of positive sentences to total sentences in review
pos = [] # number of positive [4] sentences in review
neg = [] # number of negative [0] sentences in review
posprop = [] # proportion of positive [4] sentences in review
negprop = [] # proportion of negative [0] sentences in review
# sentiment of first sentence in review
#first_sentence = []
# sentiment of last sentence in review
#last_sentence = []
movie_ids = sort(sentiments.keys())
for movie in movie_ids:
    num_sentences = 0
    sum_sentiment = 0
    num_pos4 = 0
    num_pos = 0
    num_neg = 0
    for review in sentiments[movie].itervalues():
        review = [float(val) for val in review]
        num_sentences += len(review)
        sum_sentiment += sum(review)
        num_pos4 += review.count(4)
        num_pos += review.count(4) + review.count(3)
        num_neg += review.count(0)
    avg.append(sum_sentiment/num_sentences)
    pos.append(num_pos4)
    posprop.append(float(num_pos4)/num_sentences)
    neg.append(num_neg)
    negprop.append(float(num_neg)/num_sentences)
    prop.append(float(num_pos)/num_sentences)
    
pickle.dump([avg, prop, pos, neg, posprop, negprop], open('avg_prop_pos_neg_posprop_negprop.pickle', 'wb'))
"""
# To look for missing files not processed by nlp (not necessary)
import os

f = open("reviewlists/reviewlist_COMPLETE.txt",'r')
out = f.readlines()
missing = []
    
for f in os.listdir("nlp/"):
    filename = 'reviews/' + f[:-8] + '.txt\n'
    if not filename in out:
        missing.append(f)
print missing, len(missing)
"""