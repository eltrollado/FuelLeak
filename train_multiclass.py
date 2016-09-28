import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import filesystem as fs
import math
import nn
from pybrain.utilities import percentError
from pybrain.datasets import *
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import LSTMLayer, SoftmaxLayer, SigmoidLayer, TanhLayer
from pybrain.tools.validation import Validator
import pickle


def make_zips(df):
    dff = nn.calculate_diff(df)
    labels = df['l']
    return (dff,labels)

def make_traing_dataset_zipped(frames, window=5):
    # type: (list, list, int) -> SupervisedDataSet
    ds = ClassificationDataSet(5,nb_classes=3, class_labels=['normal','leak-tank','leak-pipe'])

    for frame, label in frames:
        for i in range(len(frame.index) - window):
            sample = frame.ix[i:i+window].tolist()
            output = label.ix[i+window]
            ds.appendLinked(sample,[output])
    ds.calculateStatistics()
    ds._convertToOneOfMany()
    return ds

ds1 = fs.extract_container_data(1)
ds2 = fs.extract_container_data(2)
ds3 = fs.extract_container_data(3)


trset = map(make_zips,[ds3[3], ds3[1], ds2[2]])
tds = make_traing_dataset_zipped(trset)

tesset = map(make_zips, [ds3[2], ds2[1], ds1[1]])
testd = make_traing_dataset_zipped(tesset)

testdff = nn.calculate_diff(ds3[2])
test_data = nn.convert_to_windows(testdff)

net = buildNetwork(5,5,3,hiddenclass=LSTMLayer,recurrent=True, outclass=SoftmaxLayer, bias=True)

if False:
    print 'beginning the training'
    trainer = BackpropTrainer(net, learningrate=0.001, momentum=0.99, weightdecay=0.0002)
    errors = trainer.trainUntilConvergence(tds, maxEpochs=50)
    plt.plot(errors[0])
    plt.plot(errors[1])
    plt.figure()
    l = trainer.testOnClassData()
    print percentError(l,tds['class'])

    f = open('_learned', 'w')
    pickle.dump(net, f)
    f.close()
else:
    f = open('_learned_multi_lstm', 'r')
    net = pickle.load(f)
    f.close()


res = net.activateOnDataset(testd)
res2 = res.argmax(axis=1)
print percentError(res2, testd['class'])


lab = ds2[1]['l']

res = map(net.activate,test_data)
rng = pd.date_range('1/1/2014', '1/8/2014', freq='15Min')
e = pd.DataFrame(res, index=rng[5:])

wyk = e.join(testdff)

plt.plot(wyk)


plt.show()




