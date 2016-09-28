import pybrain
from pybrain.datasets import *
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
import pickle
import numpy as np
import math
import pandas


def convert_to_windows(ts, window=5):
    # type: (pandas.TimeSeries, int) -> list
    windows = []
    for i in range(len(ts.index) - window):
        sample = ts.ix[i:i + window]
        windows.append(sample)
    return windows

def make_traing_dataset(frames, labels, window=5):
    # type: (list, list, int) -> SupervisedDataSet
    ds = ClassificationDataSet(5,nb_classes=2, class_labels=['normal','leak'])

    for frame, label in zip(frames,labels):
        for i in range(len(frame.index) - window):
            sample = frame.ix[i:i+window].tolist()
            output = label.ix[i+window]
            ds.appendLinked(sample,[output])
    ds.calculateStatistics()
    return ds


def make_traing_dataset_zipped(frames, window=5):
    # type: (list, list, int) -> SupervisedDataSet
    ds = ClassificationDataSet(5,nb_classes=2, class_labels=['normal','leak'])

    for frame, label in frames:
        for i in range(len(frame.index) - window):
            sample = frame.ix[i:i+window].tolist()
            output = label.ix[i+window]
            ds.appendLinked(sample,[output])
    ds.calculateStatistics()
    return ds


def make_classif_traing_dataset(frames, labels, window=5):
    # type: (list, list, int) -> SupervisedDataSet
    ds = ClassificationDataSet(5,target=2, class_labels=['normal','leak'])

    for frame, label in zip(frames,labels):
        for i in range(len(frame.index) - window):
            sample = frame.ix[i:i+window].tolist()
            output = label.ix[i+window]
            ds.appendLinked(sample,[output, 1 - output])
    ds.calculateStatistics()
    return ds


def calculate_diff(df):
    # type: (pandas.DataFrame) -> pandas.TimeSeries
    df['b'] = df[1] - df[2] + df[3]
    return df['b'].diff().apply(log_scale)


def log_scale(x):
    if x > 0:
        return math.log10(x + 1)
    elif x < 0:
        return -math.log10(-x + 1)
    else:
        return 0



