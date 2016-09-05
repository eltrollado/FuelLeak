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

dataset = 3
container = 3

df = fs.extract_container_data(dataset)

lg3 = nn.calculate_diff(df[3])
lg2 = nn.calculate_diff(df[1])

tds = nn.make_traing_dataset([lg2,lg3], df['l'])

print 'beginning the training'
net = buildNetwork(5,10, 10,1, bias=True)
trainer = BackpropTrainer(net, learningrate=0.01, momentum=0.50)
trainer.trainOnDataset(tds, 50)
l = trainer.testOnData(verbose=True)

print l


