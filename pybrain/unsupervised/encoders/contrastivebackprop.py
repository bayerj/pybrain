#!/usr/bin/env python2.5
# -*- coding: utf-8 -*-


__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'


import itertools
import random

import scipy

from pybrain.unsupervised.encoders.encoder import Encoder


class ContrastiveBackprop(Encoder):

  def __init__(self):
    self.network = None

    # TODO: work these fields into configurable arguments.
    self.learnrate = 0.05 
    self.confabSteps = 10 
    self.confabStepRate = 0.05
    self.epochs = 250

  def learn(self, dataset):
    self.dataset = dataset
    for _ in xrange(self.epochs):
      self.learnEpoch()

  def learnEpoch(self):
    indexes = range(self.dataset.getLength())
    random.shuffle(indexes)
    for i in indexes:
      self.network.reset()
      self.network.resetDerivatives()
      item = self.dataset.getLinked(i)[0]
      output = self.network.activate(item)
      self.network.backActivate(output)
      derivs = -self.network.derivs[:]
      self.makeConfabulation(item)
      derivs += self.network.derivs[:] * 0.1
      self.network.params[:] = self.network.params + self.learnrate * derivs

  def makeConfabulation(self, thisitem):
    energy = self.network.activate(thisitem).sum()
    for i in itertools.count():
      item = thisitem.copy()
      item += scipy.random.standard_normal(item.shape) * 0.001
      for _ in xrange(self.confabSteps):
        self.network.reset()
        self.network.resetDerivatives()
        output = self.network.activate(item)
        inDeriv = self.network.backActivate(output)
        item += self.confabStepRate * inDeriv
      if energy > output.sum():
        break
      else:
        if i > 20:
          print "This is rejecting a lot..."
    return item

  def encode(self, item):
    return self.network.activate(item)

  def decode(self, code):
    raise NotImplementedError("No decoding for contrastive backprop.")
