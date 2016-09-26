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
from pybrain.structure.modules import LSTMLayer
import pickle



# a=  fs.get_data_set(3)
# a['pistols'].plot()
# a['refuel'].plot()
# a['tank'].plot()
# plt.show()
# exit(1)

dataset = 3
container = 2

testdataset = 3
testcontainer = 2

df = fs.extract_container_data(dataset)
df2 = fs.extract_container_data(2)


testd = nn.calculate_diff(df[1])
test_data = nn.convert_to_windows(testd)

lg3 = nn.calculate_diff(df[3])
lg1 = nn.calculate_diff(df[1])
lg2 = nn.calculate_diff(df2[2])

tds = nn.make_traing_dataset([lg1,lg3,lg2], [df[1]['l'],df[3]['l'], df2[2]['l']])


net = buildNetwork(5,5,1, hiddenclass=LSTMLayer, bias=True, recurrent=True)

try:
    f = open('_learned', 'r')
    net = pickle.load(f)
    f.close()
except:
    print 'beginning the training'
    trainer = BackpropTrainer(net, learningrate=0.01, momentum=0.10)
    trainer.trainUntilConvergence(tds, maxEpochs=100)
    l = trainer.testOnData()
    print l
    f = open('_learned', 'w')
    pickle.dump(net, f)
    f.close()

res = map(net.activate,test_data)

rng = pd.date_range('1/1/2014', '1/8/2014', freq='15Min')
e = pd.DataFrame(res, index=rng[5:])

wyk = e.join(testd)

plt.plot(wyk)


plt.show()




