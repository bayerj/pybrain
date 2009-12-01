#!/usr/bin/env python2.5
# -*- coding: utf-8 -*-

# This module is heavily inspired by django's model architecture. (See 
# http://djangoproject.org for more details on django, the best web framework
# out there!)

__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'


import itertools

from pybrain.datasets.containers import (Vectors, Scalars, Sequences, 
                                         containerRegistry)


class DataSetType(type):
  
  def __new__(cls, name, bases, attrs):
    super_new = super(DataSetType, cls).__new__
    if not any(isinstance(i, DataSetType) for i in bases):
      # Do nothing special for non-DataSetType-subclasses
      return super_new(cls, name, bases, attrs)

    # Create class.
    module = attrs.pop('__module__')
    klass = super_new(cls, name, bases, {'__module__': module})

    forbidden_handles = set(['container'])
    forb_and_used = set(attrs.keys()) & forbidden_handles
    if forb_and_used:
      raise TypeError('Forbidden field names: %s' % ", ".join(forb_and_used))

    setattr(klass, 
            '_fieldspecs', 
            sorted(attrs.items(), key=lambda f: f[1]._ticket))

    return klass


class DataSet(object):

  __metaclass__ = DataSetType

  def __init__(self, sizes, containertype='list'):
    """Create a dataset that has initialized fields for the specified sizes
    given by the iterable of integers `sizes`.
    
    The used container type can be specified by `containertype`."""
    # We have to keep this to have an ordering of the fields.
    self._fields = []
    sizes = iter(sizes)
    for name, spec in self._fieldspecs:
      if type(spec) == Scalars:
        size = 1
      else:
        size = sizes.next()
      field = containerRegistry[type(spec), containertype](size)
      self._fields.append(field)
      setattr(self, name, field)

  @classmethod
  def fromIter(cls, iters, containertype='list'):
    """Return a dataset with values initialized from the given iterators in
    `iters`. 

    The size for the fields will be infered from the values in the iterator. The
    used container type can be specified by `containertype`."""
    # Retrieve size from first items of iterators.
    if len(iters) != len(cls._fieldspecs):
      raise ValueError("Wrong number of initializing iterables supplied.")

    zipped = itertools.izip(*iters) 
    
    first = zipped.next()
    sizes = [cls._fieldDim(i, spec) 
             for i, (name, spec) in zip(first, cls._fieldspecs)]

    # Repair iterator.
    zipped = itertools.chain([first], zipped)

    # Instantiate dataset and append items from iterator.
    ds = cls(sizes, containertype)

    # Fill dataset with data.
    # TODO: enable possibility to optimize this container specifically.
    for i in zipped:
      ds.append(*i)
    return ds

  def __len__(self):
    if not self._checkSync():
      raise OutOfSyncException("DataSet has unequal lengths for fields.")
    return len(self._fields[0])

  def __iter__(self):
    if not self._checkSync():
      raise OutOfSyncException("DataSet has unequal lengths for fields.")
    return itertools.izip(*self._fields)

  def __getitem__(self, idx):
    return tuple(i[idx] for i in self._fields)

  @classmethod
  def _fieldDim(cls, initializer, spec):
    """Return the dimension of a field given by a representative item."""
    if isinstance(spec, Sequences):
      return len(initializer[0])
    elif isinstance(spec, Scalars):
      return 1
    elif isinstance(spec, Vectors):
      return len(initializer)
    else:
      raise ValueError("Unknown field type.")

  def _checkSync(self):
    """Tell whether all fields have the same length."""
    fields = iter(self._fields)
    firstlength = len(fields.next())
    return all(len(f) == firstlength for f in fields)

  def append(self, *items):
    for field, item in zip(self._fields, items):
      field.append(item)
