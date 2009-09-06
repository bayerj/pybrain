# -*- coding: utf-8 -*-
 

__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'


try:
  import json
except ImportError, e:
  try:
    import simplejson as json
  except ImportError, e:
    raise ImportError("Need simplejson or python2.6")

from pybrain.utilities import canonicClassString, multimethod


def readFromFileObject(flo):
  dct = json.load(flo)
  return fromRepresentation(dct)


def readFromFile(filename):
  return readFromFileObject(open(filename))


def writeToFileObject(ds, flo):
  dct = toRepresentation(ds)
  json.dump(dct, flo)


def writeToFile(ds, filename):
  writeToFileObject(ds, open(filename, 'w+'))


@multimethod(SequentialDataSet)
def toRepresentation(ds):
  data = {'type': 'SequentialDataSet',
          'indim': ds.indim,
          'outdim': ds.outdim,
          'sequences': []}
  for seq in ds:
    this_seq = [(s.tolist(), t.tolist()) for s, t in seq]
    data['sequences'].append(this_seq)
  return data


@multimethod(SequentialDataSet)
def fromRepresentation(dct):
  ds = SequentialDataSet(dct['indim'], dct['outdim'])
  for seq in dct['sequences']:
    ds.newSequence()
    for sample, target in seq:
      ds.addSample(sample, target)
  return ds


