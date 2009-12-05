#!/usr/bin/env python2.6
# -*- coding: utf-8 -*-


__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'


import optparse
import sys

import scipy
from pybrain.datasets import UnsupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.unsupervised.encoders import ContrastiveBackprop
from pybrain.structure import TanhLayer
import matplotlib.pyplot as plt


def make_optparse():
  parser = optparse.OptionParser()
  return parser


def make_circle():
  circle_x = scipy.linspace(-1, 1, 100)
  circle_y = scipy.sqrt(1 - circle_x**2)
  #circle_y = circle_x[:]
  #circle_x = scipy.hstack([circle_x, circle_x])
  #circle_y = scipy.hstack([circle_y, -circle_y])
  return circle_x, circle_y


def make_encoder():
  cb = ContrastiveBackprop()
  net = buildNetwork(2, 5, 1, hiddenclass=TanhLayer, fast=True)
  cb.network = net
  cb.learnrate = 0.005 
  cb.confabSteps = 5 
  cb.confabStepRate = 0.00005
  cb.epochs = 20
  return cb


def main():
  options, args = make_optparse().parse_args()

  X, Y = make_circle()
  X += scipy.random.standard_normal(X.shape) * 0.01
  Y += scipy.random.standard_normal(Y.shape) * 0.01

  # Make dataset out of that.
  ds = UnsupervisedDataSet(2)
  for item in zip(X.ravel(), Y.ravel()):
    ds.appendLinked(item)

  print "Learning...", 
  encoder = make_encoder()
  encoder.learn(ds)
  print "finished"
  
  minx, maxx = X.min(), X.max()
  miny, maxy = Y.min(), Y.max()
  sidelength = 100
  A = scipy.empty((sidelength, sidelength))
  for x in scipy.linspace(minx, maxx, sidelength):
    for y in scipy.linspace(miny, maxy, sidelength):
      A[x, y] = encoder.encode((x, y))

  # Plot circle data.
  plt.subplot(131)
  plt.plot(X, Y, 'x')

  plt.subplot(132)
  A -= A.min()
  A /= A.max()
  plt.imshow(A, cmap='gray')

  plt.subplot(133)
  plt.hist(A.ravel(), 7)

  print encoder.network.params

  plt.show()


  return 0


if __name__ == '__main__':
  sys.exit(main())

