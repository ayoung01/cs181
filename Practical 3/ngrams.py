import os
import pickle
from collections import Counter
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
from scipy import sparse

import util

from classification_starter import extract_feats, first_last_system_call_feats, system_call_count_feats

# tree = ET.parse('./train/00bee48acc9d1774e4edf96f9582fac06b2ec1f14.None.xml')
# print tree.getroot().getchildren()[0].getchildren()[0].getchildren()[0].getchildren()[0].tag
# for a in tree.iter('load_image'):
#	print a

# extract_feats([first_last_system_call_feats, system_call_count_feats])

'''
file_names = os.listdir('train')
file_names.sort()

l = []
d = {"Agent":0, "AutoRun":1, "FraudLoad":2, "FraudPack":3, "Hupigon":4, "Krap":5, "Lipler":6, "Magania":7, "None":8, "Poison":9, "Swizzor":10, "Tdss":11, "VB":12, "Virut":13, "Zbot":14}
n = 0
for datafile in file_names:
    # extract id and true class (if available) from filename
    id_str,clazz = datafile.split('.')[:2]
    l.append(d[clazz])
    print n
    n = n+1

l = np.asmatrix(np.asarray(l))
l = np.transpose(l)
f = open("y_train",'wb')
pickle.dump(l,f)
f.close()

'''

file_names = os.listdir('train')
file_names.sort()

p = len(file_names)


n = 0

ngrams_by_doc = []

for i, datafile in enumerate(file_names):
    grams = ""
    tree = ET.parse(os.path.join('train',datafile))
    in_all_section = False
    for el in tree.iter():
        # ignore everything outside the "all_section" element
        if el.tag == "all_section" and not in_all_section:
            in_all_section = True
        elif el.tag == "all_section" and in_all_section:
            in_all_section = False
        elif in_all_section:
            grams += el.tag + " "
    print i
    ngrams_by_doc.append(grams.strip())

# m = np.asmatrix(m)

#f = open("x_test",'wb')
#pickle.dump(m,f)
#f.close()




