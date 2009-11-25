#!/usr/bin/env python2.5
# -*- coding: utf-8 -*-


__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'


import scipy

from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer, LinearLayer, FullConnection, BiasUnit
from pybrain.datasets import SupervisedDataSet

try:
  from arac.pybrainbridge import _FeedForwardNetwork
except: 
  pass


class AutoEncoder(object):

  def __init__(self, indim, layers, outclass=LinearLayer, fast=False):
    """Create a new AutoEncoder.

    `indim` is an integer specifying the dimensionality of the data being
    modelled. 
    
    `layers` is a list of (layer class, dim) tuples that the 
    autoencoder is built from. This will be built in a mirroring manner, e.g.
    layers [(A, 10), (B, 5), (C, 2)] will result in a structure like

      InputLayer -> A(10) -> B(5) -> C(2) -> B(5) -> A(10) -> OutputLayer

    where C is assumed to be the code layer as it is the last in the row.
    
    The input layer is always a LinearLayer. The output layer type can be
    specified by `outclass` which defaults to LinearLayer.

    If `fast` is set to True, fast networks will be used.
    """
    self.indim = indim

    # Build up network.
    n = _FeedForwardNetwork() if fast else FeedForwardNetwork()

    previous = LinearLayer(indim, name='inpt')
    n.addInputModule(previous)
    
    bias = BiasUnit()
    n.addModule(bias)

    # Add layers up until coding layer.
    for klass, dim in layers:
      layer = klass(dim)
      con = FullConnection(previous, layer)
      n.addConnection(con)
      n.addModule(layer)
      previous = layer

      bcon = FullConnection(bias, layer)
      n.addConnection(bcon)

    previous.name = 'code'


    # Add layers for decoding. 
    for klass, dim in list(reversed(layers[:-1])) + [(outclass, indim)]:
      layer = klass(dim)
      con = FullConnection(previous, layer)
      n.addConnection(con)
      n.addModule(layer)
      previous = layer

      bcon = FullConnection(bias, layer)
      n.addConnection(bcon)

    n.addOutputModule(previous)
    n.sortModules()
    self.network = n

  def learn(self, dataset, learningrate=0.005, momentum=0.0):
    # For normalization.
    self.maxis = dataset.data['sample'][:dataset.getLength()].max(axis=0)
    self.minis = dataset.data['sample'][:dataset.getLength()].min(axis=0)

    if dataset.getDimension('sample') != self.indim:
      raise ValueError("Dataset does not fit specified input dimension.")
    # Construct dataset with input == target.
    sds = SupervisedDataSet(self.indim, self.indim)
    for i in xrange(len(dataset)):
      sample = dataset.getLinked(i)[0]
      sample -= self.minis
      sample /= self.maxis - self.minis
      sds.appendLinked(sample, sample)
    self.network.randomize()
    trainer = BackpropTrainer(self.network, sds, 
                              learningrate=learningrate, momentum=momentum)
    trainer.trainUntilConvergence(maxEpochs=10)

  def encode(self, inpt):
    self.network.reset()
    this_input = (inpt - self.minis) / (self.maxis - self.minis)
    self.network.activate(this_input)
    return self.network['code'].outputbuffer[0].copy()

  def decode(self, code):
    self.network.reset()
    self.network['code'].outputbuffer[0][:] = code
    res = self.network.activate([-1] * self.indim) * (self.maxis - self.minis)
    res += self.minis()
    return 
