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

networks = ['_learned_deep_tanh', '_learned_deep_2tanh', '_learned_lstm_tanh', '_learned_recurrent_tanh', '_learned_lstm2', '_learned_ff_tanh', '_learned_recurrent']


def test_networks(test_sets):
    net = None

    for name in networks:
        try:
            f = open(name, 'r')
            net = pickle.load(f)
            f.close()
        except:
            print 'network {} failed to load'.format(name)
            continue

        scores = []
        for test_set, test_labels in test_sets:
            inputs = nn.convert_to_windows(test_set)
            res = map(net.activate, inputs)
            res2 = map(lambda x: 1 if x > 0.5 else 0, res)
            per = Validator.classificationPerformance(res2, test_labels[5:])
            scores.append(per)

        print name
        print scores
        print sum(scores)/len(scores)


def make_zips(df):
    dff = nn.calculate_diff(df)
    labels = df['l']
    return (dff,labels)

ds1 = fs.extract_container_data(1)
ds2 = fs.extract_container_data(2)
ds3 = fs.extract_container_data(3)



testd = nn.calculate_diff(ds3[2])
test_data = nn.convert_to_windows(testd)
lab = ds3[2]['l']

test_set = map(make_zips,[ds1[1],ds2[1],ds3[4]])
test_networks(test_set)

exit(1)