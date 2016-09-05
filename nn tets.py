# learn XOR with a nerual network with saving of the learned paramaters

import pybrain
from pybrain.datasets import *
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
import pickle
import numpy as np
import matplotlib.pyplot as plt


if True:
    ds = SupervisedDataSet(2, 1)
    ds.addSample((0, 0), (0))
    ds.addSample((0, 1), (0))
    ds.addSample((1, 0), (0))
    ds.addSample((1, 1), (0))
    ds.addSample((1, 0.5), (1))



    net = buildNetwork(2,16, 16, 1, bias=True)

    try:
        f = open('_learned', 'r')
        net = pickle.load(f)
        f.close()
    except:
        trainer = BackpropTrainer(net, learningrate=0.01, momentum=0.99)
        trainer.trainOnDataset(ds, 1000)
        trainer.testOnData()
        f = open('_learned', 'w')
        pickle.dump(net, f)
        f.close()

    print net.activate((1, 1))

    xx, yy = np.meshgrid(np.arange(0, 1, 0.1),
                         np.arange(0, 1, 0.1))

    d = zip(xx.ravel(), yy.ravel())
    Z = map(net.activate,d)

    Z = np.asarray(Z)
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z,)
    plt.axis('off')
    plt.show()
    #
    # # Plot also the training points
    # plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=pl.cm.Paired)