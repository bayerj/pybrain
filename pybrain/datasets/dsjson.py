# -*- coding: utf-8 -*-
 
__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'


import json

from pybrain.datasets import SequentialDataSet


def serialize(ds, filename):
  if isinstance(ds, SequentialDataSet):
    return serialize_sequential(ds, filename)
  else:
    raise ValueError("Unknown dataset type for JSON-serialization, %s" % type(ds))


def unserializeSequential(filename):
  data = json.load(file(filename))
  ds = SequentialDataSet(data['indim'], data['outdim'])
  for seq in data['sequences']:
    ds.newSequence()
    for sample, target in seq:
      ds.addSample(sample, target)
  return ds


def serializeSequential(ds, filename):
  # Built up dictionary for serialization.
  data = {'type': 'SequentialDataSet',
          'indim': ds.indim,
          'outdim': ds.outdim,
          'sequences': []}
  for seq in ds:
    this_seq = [(s.tolist(), t.tolist()) for s, t in seq]
    data['sequences'].append(this_seq)

  json.dump(data, file(filename, 'w'))
