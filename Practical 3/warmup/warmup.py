import csv
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from scipy import optimize as opt

"""
===============================================================
INPUT FILE PROCESSING
===============================================================
"""
f = open('fruit.csv','rb')
reader = csv.reader(f)
headers = reader.next()
column = {} #{'fruit':[1,1,...], 'width':[8.4, 8,...], 'height':[7.3, 6.8,...]}
for h in headers:
    column[h] = []
for row in reader:
    for h, v in zip(headers, row):
        column[h].append(float(v))
n = len(column['width'])
phi = np.concatenate((
    np.array([1.] * n)[:,np.newaxis],
    np.array(column['width'])[:,np.newaxis],
    np.array(column['height'])[:,np.newaxis],
    ), axis=1)
phi2 = np.concatenate((
    np.array(column['fruit'])[:,np.newaxis],
    np.array(column['width'])[:,np.newaxis],
    np.array(column['height'])[:,np.newaxis],
    ), axis=1)
X = phi[:,1:]
fruits = []
for fruit in column['fruit']:
    if fruit==1: #apples
        fruits.append([1,0,0])
    if fruit==2: #oranges
        fruits.append([0,1,0])
    if fruit==3: #lemons
        fruits.append([0,0,1])
t = np.array(fruits)
N = phi.shape[0]; M = phi.shape[1]; K = t.shape[1];

"""
===============================================================
MULTICLASS LOGISTIC REGRESSION
===============================================================
"""
# Calculate activation
def a(w, k, n, phi):
    return np.exp(np.dot(w[k], phi[n]));

# Calculate probability of class k given data
def y(w, k, n, phi):
    activation_sum = 0
    for x in xrange(K):
        activation_sum += a(w, x, n, phi)
    return a(w, k, n, phi)/activation_sum

"""
Cross-entropy error function for multiclass classification
See Bishop(p.209)

@type w, t, phi: numpy.ndarray
@param w: K x M vector of weight parameters
@param t: N x K matrix of target variables
@param phi: N x M design matrix

@rtype: number
@return: error value
"""
def neg_ll(w, t, phi):
    #print "ll", w
    w = w.reshape(3,3)
    error = 0    
    for n in xrange(N):
        for k in xrange(K):
            y_nk = y(w, k, n, phi)
            error -= t[n][k] * np.log(y_nk)
    return error

"""
Gradient of the error function

@type w, t, phi: numpy.ndarray
@param: w = K x M vector of weight parameters
@param: t = N x K matrix of target variables
@param: phi = N x M design matrix

@rtype: numpy.ndarray
@return: K x M matrix (in 1-D array)
"""
def grad_ll(w, t, phi):
    #print "grad", w
    grad = []
    w = w.reshape(3,3)
    for j in xrange(M):
        x = np.array([0.,0.,0.])
        for n in xrange(N):
            y_nj = y(w, j, n, phi)
            x += np.dot(y_nj - t[n][j], phi[n])
        grad.extend(x)
    return np.array(grad)

# Determine optimal weight parameters
opt = opt.fmin_bfgs(f=neg_ll, x0=[0]*9, fprime=grad_ll, args=(t, phi)).reshape(3,3)

def logistic_model(phi):
    a = []
    for n in xrange(phi.shape[0]):
        phi_n = phi[n]
        pred = []
        for k in xrange(K):
            y_nj = np.exp(np.dot(opt[k], phi_n))
            pred.append(y_nj)
        a.append(pred.index(max(pred))) # append predicted type
    return np.array(a)

def plot_logistic():
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .01),
                        np.arange(y_min, y_max, .01))
    
    Z = logistic_model(np.c_[[1]*len(xx.ravel()), xx.ravel(), yy.ravel()]) 
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    
    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=phi2[:,0])
    plt.show()

"""
===============================================================
GENERATIVE CLASSIFIER W/ GAUSSIAN CLASS-CONDITIONAL BOUNDARIES
===============================================================
"""
# Calculate ML priors:
priors = []; N_k = [];
for k in xrange(K):
    sub = phi2[phi2[:,0] == k+1] # subset by class
    N_k.append(float(len(sub)))
    priors.append(N_k[k]/N)
priors = np.array(priors) # array([ 0.3220339 ,  0.40677966,  0.27118644])
N_k = np.array(N_k) # array([ 19.,  24.,  16.])

# Calculate ML means:
mu = [] # 0: apples, 1: oranges, 2: lemons
for k in xrange(K):
    sub = phi2[phi2[:,0] == k+1]
    mu.append(np.mean(sub[:,1]))
    mu.append(np.mean(sub[:,2]))
mu = np.array(mu).reshape(K,M-1) # K x M-1 matrix

# Calculate covariance matrix
S = [] # contains [sigma0, sigma1, sigma2]
for k in xrange(K):
    tot = np.zeros((M-1,M-1))
    for n in xrange(N):
        if phi2[n,0] == k+1: # for only n \in C_k
            dif = np.array(X[n][:,np.newaxis] - mu[k][:,np.newaxis])
            tot += np.dot(dif, dif.T)
    S.append(tot/N_k[k])
sigma = np.zeros(S[0].shape)
for k in xrange(K):
    sigma += N_k[k]/N*S[k]

w_0 = []; w_k = [];
for k in xrange(K):
    w_k.append(np.dot(inv(sigma), mu[k].reshape(2,1)))
    w_0.append(-.5*np.dot(mu[k], np.dot(inv(sigma), mu[k].reshape(2,1)))[0]+np.log(priors[k]))

def generative_model(phi):
    a = []
    for n in xrange(phi.shape[0]):
        phi_n = phi[n]
        pred = []
        for k in xrange(K):
            y_nj = np.exp(np.dot(w_k[k].T, phi_n)+w_0[k])
            pred.append(y_nj)
        a.append(pred.index(max(pred))) # append predicted type
    return np.array(a)

def plot_gen():
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .01),
                        np.arange(y_min, y_max, .01))
    Z = generative_model(np.c_[xx.ravel(), yy.ravel()]) 
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    #plt.axis('off')
    
    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=phi2[:,0])
    plt.show()
    
plot_logistic()
#plot_gen()

# Count the number of correctly classified cases:
counts = [0,0,0]
for i, n in enumerate(generative_model(X)):
    if n+1 == phi2[i][0]:
        counts[n]+=1
        
counts = [0,0,0]
for i, n in enumerate(logistic_model(phi)):
    if n+1 == phi2[i][0]:
        counts[n]+=1