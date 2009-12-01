#!/usr/bin/env python2.5
# -*- coding: utf-8 -*-


__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'


import collections
import itertools
import os
import struct
import tempfile
import threading

import scipy
from scipy.io.numpyio import fread, fwrite


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

  doublesize = struct.calcsize('d')

  def __init__(self, dim):
    self.dim = dim
    self.itemsPerFile = self.max_doubles_per_file / dim
    self.files = []

  def __del__(self):
    for f in self.files:
      os.remove(f)

  def __getitem__(self, idx):
    # Determine the file the vector resides in.
    fileidx, offset = divmod(idx, self.itemsPerFile)
    filename = self.files[fileidx]
    with open(filename) as fd:
      fd.seek(offset * self.doublesize)
      item = fread(fd, self.dim, 'd')
      return item

  def fileForAppend(self, needed):
    filesize = os.path.getsize(self.files[-1]) if self.files else float('inf')
    if filesize + needed * self.doublesize >= self.max_file_size:
      # We need to create a new file if the previous file is approaching the
      # upper file limit.
      _fd = tempfile.NamedTemporaryFile('w', delete=False)
      fd = open(_fd.name, 'r+')
      self.files.append(fd.name)
    else:
      fd = open(self.files[-1], 'r+')
      fd.seek(0, 2)
    return fd

  def append(self, item):
    item = scipy.asarray(item, dtype='float64')
    with self.fileForAppend(self.dim) as fd:
      fwrite(fd, self.dim, item)
    print os.path.getsize(self.files[-1])


class ExternalScalarsContainer(ExternalVectorsContainer):

  def __init__(self):
    super(ExternalScalarsContainer, self).__init__(1)

  def __getitem__(self, idx):
    super(ExternalVectorsContainer, self)[idx][0]

  def append(self, item):
    super(ExternalVectorsContainer, self).append([item])


class ExternalSequencesContainer(ExternalVectorsContainer):

  @property
  def fill(self):
    return len(self.sequenceToFiles)

  def __init__(self, dim):
    super(ExternalSequencesContainer, self).__init__(dim)

    # Mapping of a sequence to its file index.
    self.sequenceToFiles = []     

    # Mapping of a sequence to its lengths.
    self.sequenceToLengths = []     
    
    # Mapping of a filename to the sequence indices that it contains.
    self.fileToSequences = collections.defaultdict(lambda: [])

  def __getitem__(self, idx):
    fileidx = self.sequenceToFiles[idx]
    filename = self.files[fileidx]
    with open(filename, 'r') as fp:
      # Determine the indexes of the files that are in the same file before this
      # sequence.
      precedSeqs = [i for i in self.fileToSequences[fileidx] if i < idx]
      print self.fileToSequences
      print "Predeceeding sequences", precedSeqs
      # Calculate the offset in the file.
      seq_offset = sum(self.sequenceToLengths[i] * self.dim for i in precedSeqs)
      # Seek to that offset...
      fp.seek(self.doublesize * seq_offset)
      # ... and return the corresponding sequence.
      res = fread(fp, self.sequenceToLengths[idx] * self.dim, 'd')
      print "reading", res, "from", self.doublesize * seq_offset
    res.shape = res.size / self.dim, self.dim

    return res

  def append(self, item):
    item = scipy.asarray(item, dtype='float64')
    with self.fileForAppend(item.size) as fd:
      print "writing", item, "to", fd.tell()
      fwrite(fd, item.size, item)

    fileidx = len(self.files) - 1
    self.sequenceToFiles.append(fileidx)
    self.sequenceToLengths.append(item.shape[0])
    self.fileToSequences[fileidx].append(len(self.sequenceToFiles) - 1)


containerRegistry = {
  (Vectors, 'numpy'): NumpyVectorsContainer,
  (Scalars, 'numpy'): NumpyScalarsContainer,
  (Sequences, 'numpy'): NumpySequencesContainer,
  
  (Vectors, 'list'): lambda _: list(),
  (Scalars, 'list'): lambda _: list(),
  (Sequences, 'list'): lambda _: list(),
  
  (Vectors, 'external'): ExternalVectorsContainer,
  (Scalars, 'external'): ExternalScalarsContainer,
  (Sequences, 'external'): ExternalSequencesContainer,
}





