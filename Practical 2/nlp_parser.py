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