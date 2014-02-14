import numpy as np
import scipy.spatial
import cPickle

# Define the number of clusters K
K = 16

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

batch1 = unpickle('data_batch_1')
batch2 = unpickle('data_batch_2')
batch3 = unpickle('data_batch_3')
batch4 = unpickle('data_batch_4')
batch5 = unpickle('data_batch_5')

# combine 5 batches to get the array of 50000 data vectors
v = np.concatenate((batch1["data"],batch2["data"]))
v = np.concatenate((v,batch3["data"]))
v = np.concatenate((v,batch4["data"]))
v = np.concatenate((v,batch5["data"]))

# normalize vector entries to be in the range [0,1]
v = v/255.0

# responsibility vector
r = np.zeros((50000,K), dtype = int)

# K mean vectors
m = np.zeros((K,3072))

# randomly initialize r to start
random = np.random.randint(K, size=50000)
for i in range(50000): 
    r[i][random[i]] = 1
    print i

print "Initialzation done!"

def compute_mean():
    v_m = np.matrix(v)
    r_m = np.matrix(r)
    for i in range(K):
        n = 0 # number of data in cluster i
        for j in range (50000):
            n = n + r[j][i]
        m[i] = np.array(np.transpose((np.transpose(v_m) * r_m[:,i]))) / n

def update_cluster():
    change = False
    distances = scipy.spatial.distance.cdist(v, m, 'sqeuclidean')
    for i in range(50000):
        p = np.argmin(distances[i])
        if r[i][p] != 1:
            r[i] = np.zeros(K)
            r[i][p] = 1
            change = True
    return change

keep_going = True

count = 0

while keep_going == True:
    compute_mean()
    keep_going = update_cluster()
    count = count + 1
    print count

print m

np.save("cluster_centers", m)
np.save("responsibility_vectors", r)