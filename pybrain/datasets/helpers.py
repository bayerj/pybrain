#!/usr/bin/env python2.5
# -*- coding: utf-8 -*-


__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'


import scipy

from pybrain.datasets.containers import Sequences, Vectors, Scalars


def replaceNansByMeans(dataset):
  for field, (_, spec) in zip(dataset._fields, dataset._fieldspecs):
    if isinstance(spec, (Scalars, Vectors)):
      _replaceNansByMeansVectors(field)
    elif isinstance(spec, Sequences):
      _replaceNansByMeansSequences(field)
    else:
      raise ValueError("Unknown field type: %s" % type(field))
    

def _replaceNansByMeansSequences(field):
  def items():
    for seq in field:
      for item in seq:
        yield item
  items = list(items())

  # Specialization for list containers.
  if type(field) is list:
    dim = len(items[0])
  else:
    dim = field.dim

  for d in xrange(dim):
    nonans = [i[d] for i in items if not scipy.isnan(i[d])]
    all = sum(nonans)
    mean = all / len(nonans)
    for idx, val in enumerate(items):
      if scipy.isnan(val[d]):
        items[idx][d] = mean

def _replaceNansByMeansVectors(field):
  for d in xrange(field.dim):
    nonans = [i[d] for i in field if not scipy.isnan(i[d])]
    all = sum(vals)
    mean = all / len(vals)
    for idx, val in enumerate(field):
      if scipy.isnan(val):
        field[i][d] = mean
