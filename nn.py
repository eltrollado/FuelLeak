import pybrain
from pybrain.datasets import *
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
import pickle
import numpy as np
import math


def make_traing_dataset(frames, labels, window=5):
    #type: (list, object, int) -> object
    ds = SupervisedDataSet(5,1)
    for dff in frames:
        for i in range(len(dff.index)- window):
            sample = dff.ix[i:i+window].tolist()
            label = labels.ix[i+window]
            ds.addSample(sample,(label))
    return ds


def calculate_diff(df):
    df['b'] = df[1] - df[2] + df[3]
    return df['b'].diff().apply(log_scale)


def log_scale(x):
    if x > 0:
        return math.log10(x +1)
    elif x < 0:
        return -math.log10(-x +1)
    else:
        return 0

