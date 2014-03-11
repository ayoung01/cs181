# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 15:10:41 2014

@author: vincentli2010
"""
import numpy as np
import pylab as pl

from sklearn.preprocessing import StandardScaler

X = np.load(open('x_train3', 'rb'))
y = np.array(np.load(open('y_train', 'rb'))).flatten()

# Transformation
X = np.log(X + 1)
X = StandardScaler().fit_transform(X)

X_train, X_test = X[:2468], X[2468:]
y_train, y_test = y[:2468], y[2468:]

from pybrain.datasets import ClassificationDataSet
trndata = ClassificationDataSet(X_train.shape[1], 1, nb_classes=15)
for i in xrange(X_train.shape[0]):
    trndata.addSample(X_train[i,:], y_train[i])

tstdata = ClassificationDataSet(X_test.shape[1], 1, nb_classes=15)
for i in xrange(X_test.shape[0]):
    tstdata.addSample(X_test[i,:], y_test[i])

trndata._convertToOneOfMany()
tstdata._convertToOneOfMany()

"""
print "Number of training patterns: ", len(trndata)
print "Input and output dimensions: ", trndata.indim, trndata.outdim
print "First sample (input, target, class):"
print trndata['input'][0], trndata['target'][0], trndata['class'][0]
"""


from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import SoftmaxLayer
fnn = buildNetwork(trndata.indim, 70, trndata.outdim,
                   outclass=SoftmaxLayer )

from pybrain.supervised.trainers import BackpropTrainer
trainer = BackpropTrainer(fnn, dataset=trndata,
                          momentum=0.1, verbose=True,
                          weightdecay=0.001)

# 0.8801 Hidden Units: 70 	 WeightDecay: 0.001 	 95 epochs
print '======================='
print "Hidden Units: 70 \t WeightDecay: 0.001 \t 95 epochs"
print '======================='
TRAINRESULT = []
TESTRESULT = []
from pybrain.utilities import percentError
for i in range(95):
    trainer.trainEpochs(1)
    trnresult = percentError(trainer.testOnClassData(),
                             trndata['class'] )
    tstresult = percentError(trainer.testOnClassData(dataset=tstdata),
                             tstdata['class'] )
    print "epoch: %4d" % trainer.totalepochs, \
          "  train error: %5.2f%%" % trnresult, \
          "  test error: %5.2f%%" % tstresult
    TRAINRESULT.append(trnresult)
    TESTRESULT.append(tstresult)



y_test = tstdata['class'].flatten()

y_pred = fnn.activateOnDataset(tstdata)
y_pred = y_pred.argmax(axis=1)

model = 'Network'

miss = np.zeros((15,15), dtype=int)
for i in xrange(len(y_test)):
    if y_test[i] != y_pred[i]:
        if y_test[i] < y_pred[i]:
            miss[int(y_test[i]), int(y_pred[i])] += 1
        else:
            miss[int(y_pred[i]), int(y_test[i])] += 1
import matplotlib.pyplot as plt
target_names =  ['Agent', 'AutoRun',
                 'FraudLoad', 'FraudPack',
                 'Hupigon', 'Krap',
                 'Lipler', 'Magania',
                 'None', 'Poison',
                 'Swizzor', 'Tdss',
                 'VB', 'Virut', 'Zbot']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(miss, interpolation='nearest')
fig.colorbar(cax)
plt.xticks(range(len(target_names)), target_names, rotation=45)
plt.yticks(range(len(target_names)), target_names)
plt.title(model)
plt.show()
plt.savefig('miss/' + model + '.png')

