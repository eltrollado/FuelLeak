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
import pickle

networks = ['_learned_deep_tanh', '_learned_lstm_tanh', '_learned_recurrent_tanh', '_learned_lstm2', '_learned_deep', '_learned_recurrent']
net = None

try:
    f = open('_learned_ff_tanh', 'r')
    net = pickle.load(f)
    f.close()
except:
    print 'Could not load the network'
    exit(1)

dataset = 3
tank = 2

ds = fs.extract_container_data(dataset)

testd = nn.calculate_diff(ds[tank])
test_data = nn.convert_to_windows(testd)

lab = ds[tank]['l']
res = map(net.activate,test_data)

res2 = map(lambda x: 1 if x>0.5 else 0, res)


rng = pd.date_range('1/1/2014', '1/8/2014', freq='15Min')
e = pd.DataFrame(res, index=rng[5:])

wyk = e.join(testd)

plt.plot(wyk)


plt.show()