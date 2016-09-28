import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import filesystem as fs
import math
import nn
import pybrain
from pybrain.datasets import *
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import LSTMLayer, SoftmaxLayer, SigmoidLayer, TanhLayer
from pybrain.tools.validation import Validator

import pickle


# a=  fs.get_data_set(3)
# a['pistols'].plot()
# a['refuel'].plot()
# a['tank'].plot()
# plt.show()
# exit(1)


def make_zips(df):
    dff = nn.calculate_diff(df)
    labels = df['l']
    return (dff,labels)

ds1 = fs.extract_container_data(1)
ds2 = fs.extract_container_data(2)
ds3 = fs.extract_container_data(3)

# plt.plot(ds2[1])
# plt.figure()
# plt.plot(nn.calculate_diff(ds2[1]))
# plt.show()
# exit(1)


trset = map(make_zips,[ds3[3], ds3[1], ds2[2]])
tds = nn.make_traing_dataset_zipped(trset)

testd = nn.calculate_diff(ds3[2])
test_data = nn.convert_to_windows(testd)
lab = ds3[2]['l']


# df = fs.extract_container_data(3)
# df2 = fs.extract_container_data(2)
#
# lg3 = nn.calculate_diff(df[3])
# lg1 = nn.calculate_diff(df[1])
# lg2 = nn.calculate_diff(df2[2])
#
# tds = nn.make_traing_dataset([lg1,lg3,lg2], [df[1]['l'],df[3]['l'], df2[2]['l']])


net = buildNetwork(5,5,1,hiddenclass=TanhLayer, outclass=TanhLayer, bias=True)


print 'beginning the training'
trainer = BackpropTrainer(net, learningrate=0.001, momentum=0.99, weightdecay=0.0002)
errors = trainer.trainUntilConvergence(tds, maxEpochs=50)
plt.plot(errors[0])
plt.plot(errors[1])
plt.figure()
l = trainer.testOnData()
print l
f = open('_learned', 'w')
pickle.dump(net, f)
f.close()


res = map(net.activate,test_data)

res2 = map(lambda x: 1 if x>0.5 else 0, res)
lab = ds2[1]['l']
per = Validator.classificationPerformance(res2,lab[5:])


print per

rng = pd.date_range('1/1/2014', '1/8/2014', freq='15Min')
e = pd.DataFrame(res, index=rng[5:])

wyk = e.join(testd)

plt.plot(wyk)


plt.show()




