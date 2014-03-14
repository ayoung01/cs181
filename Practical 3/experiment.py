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
'''
file_names = os.listdir('train')
file_names.sort()

x = np.zeros([3086,5])

def time_interval(t1,t2):
    m1, s1 = t1.split(':')[:2]
    m1 = float(m1)
    s1 = float(s1)
    m2, s2 = t2.split(':')[:2]
    m2 = float(m2)
    s2 = float(s2)
    return (60*m2+s2)-(60*m1+s1)

n = 0

for datafile in file_names:
    tree = ET.parse(os.path.join('train',datafile))
    n_p = 0 # total number of processes
    n_t = 0 # total number of threads
    n_f = 0 # total number of function calls
    fs = 0 # total file size
    et = 0.0 # total execution time in seconds
    processes = tree.getroot().getchildren()
    n_p = len(processes)
    for i in range(n_p):
    	p = processes[i]
        d_p = p.attrib
        if time_interval(d_p['starttime'], d_p['terminationtime']) > 0:
            et = et + time_interval(d_p['starttime'], d_p['terminationtime'])
        fs = fs + int(d_p['filesize'])
        threads = p.getchildren()
        n_t = n_t + len(threads)
        for j in range(len(threads)):
        	t = threads[j]
        	function_calls = t.getchildren()
        	n_f = n_f + len(function_calls)
    x[n][0] = n_p
    x[n][1] = n_t
    x[n][2] = n_f
    x[n][3] = fs
    x[n][4] = et
    print n
    n = n+1

f = open("additional_features",'w')
pickle.dump(x,f)
f.close()
'''

'''
f1 = open("additional_features",'r')
a = pickle.load(f1)
f1.close()

for i in range(3086):
	for j in range(5):
		if a[i][j] < 0:
			print i, j
'''
'''
file_names = os.listdir('train')
file_names.sort()
print file_names[545]
print file_names[858]
print file_names[1146]
print file_names[1753]
print file_names[2863]
'''
'''

f1 = open("additional_features",'r')
a = pickle.load(f1)
f1.close()

f2 = open("x_train",'r')
b = pickle.load(f2)
f2.close()

b = np.asarray(b)

c = np.append(b,a,axis=1)

c = np.asmatrix(c)

f = open("x_train3",'wb')
pickle.dump(c,f)
f.close()
'''

f1 = open("sorted_common_calls",'r')
a = pickle.load(f1)
f1.close()

print a