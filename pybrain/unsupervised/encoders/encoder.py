#!/usr/bin/env python2.5
# -*- coding: utf-8 -*-


__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'


import scipy


class Encoder(object):

  def reconstruction_error(self, dataset):
    """Return the sum-of-square error an encoder is doing on the given
    dataset."""
    e = 0
    for i in xrange(dataset.getLength()):
      sample = dataset.getLinked(i)[0]
      code = self.encode(sample)
      recons = self.decode(code)
      error = sample - recons
      e += scipy.dot(error, error)
    e /= dataset.getLength()
    return e
