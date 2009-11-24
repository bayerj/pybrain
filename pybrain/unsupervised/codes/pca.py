#!/usr/bin/env python2.6
# -*- coding: utf-8 -*-


__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'


import scipy

from pybrain.auxiliary.pca import pca


class PcaCode(object):

  def __init__(self, indim, latentDim):
    self.indim = indim
    self.latentDim = latentDim
    self.encoding = scipy.empty((latentDim, indim))
    self.decoding = scipy.empty((indim, latentDim))
    self.mean = scipy.empty(indim)

  def learn(self, dataset):
    if dataset.getDimension('sample') != self.indim:
      raise ValueError("Dataset does not fit specified input dimension.")
    self.data = dataset.getField('sample')[:]

    self.mean[:] = self.data.mean(axis=0)
    self.encoding[:] = scipy.asarray(pca(self.data, self.latentDim))
    self.decoding[:] = scipy.linalg.pinv2(self.encoding)

  def encode(self, inpt):
    inpt = scipy.asarray(inpt) - self.mean
    return scipy.dot(self.encoding, inpt)

  def decode(self, code):
    code = scipy.asarray(code)
    return scipy.dot(self.decoding, code) + self.mean
