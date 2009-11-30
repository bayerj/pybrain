#!/usr/bin/env python2.5
# -*- coding: utf-8 -*-


__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'


import itertools
import struct
import threading

import scipy


class FieldType(object): 

  # To save the order in which fields have been declared in classes, each Type
  # requests a number by which it can be sorted. We will use this iterator that 
  # is fetched a new number from any Type is created. 
  _field_id_lock = threading.Lock()
  _field_ids= itertools.count()

  def __init__(self):
    with self._field_id_lock:
      self._ticket = self._field_ids.next()

class Vectors(FieldType): pass
class Scalars(FieldType): pass
class Sequences(FieldType): pass


class NumpyVectorsContainer(object):

  @property
  def data(self):
    return self._data[:self.fill]

  @property
  def capacity(self):
    return self._data.shape[0]

  def __init__(self, dim):
    """Create a NumpyVectorsContainer object.

    `dim` specifies the dimensionality of every item and has to equal for
    each."""
    self.dim = dim
    self._data = scipy.empty((128, dim))
    self.fill = 0

  def __getitem__(self, idx):
    return self.data[idx]

  def changeLength(self, length):
    """Change the length of the data to `length`."""
    # Data array is filled and we have to increase the size.
    new_data = scipy.empty((length, self.dim))
    new_data[:self.fill] = self.data
    self._data = new_data
    
  def append(self, item):
    """Append an item to the container."""
    if self.freeRows() <= 0:
      self.changeLength(self.capacity * 2)
    self._data[self.fill][:] = item
    self.fill += 1

  def finalize(self):
    """Free any superfluous memory."""
    self.changeLength(self.fill)

  def freeRows(self):
    return max(self.capacity - self.fill, 0)


class NumpyScalarsContainer(NumpyVectorsContainer):

  def __init__(self):
    """Create a NumpyScalarsContainer object."""
    super(NumpyVectorsContainer, self).__init__(1)

  def append(self, item):
    """Append a scalar to the container."""
    if self.freeRows() <= self.capacity:
      self.changeLength(self.capacity)
    self._data[self.fill] = item
    self.fill += 1


class NumpySequencesContainer(NumpyVectorsContainer):

  def __init__(self, dim):
    """Create a NumpySequencesContainer object."""
    super(NumpySequencesContainer, self).__init__(dim)
    self.sequenceStarts = []

  def __getitem__(self, idx):
    start = self.sequenceStarts[idx]
    try: 
      stop = self.sequenceStarts[idx + 1]
    except IndexError:
      stop = self.fill
    return self.data[start:stop]

  def append(self, sequence):
    """Append a sequence to the container."""
    seqlength = len(sequence)
    while self.freeRows() < seqlength:
      self.changeLength(self.capacity * 2)
    self._data[self.fill:self.fill + seqlength] = sequence
    self.sequenceStarts.append(self.fill)
    self.fill += seqlength


class ExternalVectorsContainer(object):

  # Allow files up to this size in bytes.
  max_file_size = 128 * 1024 * 1024

  # Allow so many doubles per file.
  max_doubles_per_file = max_file_size / 8

  # Maximum number of simultaneously opened files.
  max_files_open = 16

  doublesize = struct.calcsize('d')

  def __init__(self, dim):
    self.dim = dim
    self.itemsPerFile = max_doubles_per_file / dim
    self.files = []

  def __getitem__(self, idx):
    # Determine the file the vector resides in.
    fileidx, offset = divmod(idx, self.itemsPerFile)
    filename = self.files[fileidx]
    with open(filename) as fp:
      fp.seek(offset * self.doublesize)
      buffer = fp.read(self.dim * self.doublesize)
    item = scipy.frombuffer(buffer, 'float64')
    return item

  def append(self, item):
    # Check size of last file.
    if not os.path.getsize(files[-1]) < self.max_file_size:
      # We need to create a new file.
      # I stopped working right here... ..
      # ....




containerRegistry = {
  (Vectors, 'numpy'): NumpyVectorsContainer,
  (Scalars, 'numpy'): NumpyScalarsContainer,
  (Sequences, 'numpy'): NumpySequencesContainer,
  
  (Vectors, 'list'): lambda _: list(),
  (Scalars, 'list'): lambda _: list(),
  (Sequences, 'list'): lambda _: list(),
}





