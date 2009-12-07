#!/usr/bin/env python2.5
# -*- coding: utf-8 -*-


__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'


import collections
import itertools
import os
import struct
import tempfile
import threading

try: 
  import multiprocessing # For python 2.6
except ImportError:
  try:
    import pyprocessing as multiprocessing # For python < 2.5
  except ImportError:
    # Fail silently.
    pass

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


class Container(object):

  def __iter__(self):
    for i in xrange(len(self)):
      yield self[i]


class ArrayContainer(Container):
  """Abstract class that is used to implement containers accessing their items 
  via serialized (= 1 dimensional) arrays."""

  @property
  def capacity(self):
    raise NotImplemented("Must be implemented by subclass.")

  def __init__(self, dim):
    self.dim = dim
    self.fill = 0
    self._data = None
    self.allocate(128)

  def allocate(self, length):
    raise NotImplemented("Must be implemented by subclass.")

  def __getitem__(self, idx):
    if idx >= self.fill:
      raise IndexError("Container index out of range.")
    return self._data[idx * self.dim:(idx + 1) * self.dim]

  def append(self, item):
    """Append an item to the container."""
    if self.freeRows() <= 0:
      self.allocate(self.capacity * 2)
    self._data[self.fill * self.dim:(self.fill + 1) * self.dim] = item
    self.fill += 1

  def finalize(self):
    """Free any superfluous memory."""
    self.allocate(self.fill)

  def freeRows(self):
    return max(self.capacity - self.fill, 0)


class NumpyVectorsContainer(ArrayContainer):

  @property
  def data(self):
    if self._data is not None:
      return self._data[:self.fill * self.dim]
    else:
      return None 

  @property
  def capacity(self):
    return self._data.shape[0]

  def allocate(self, length):
    """Change the length of the data to `length`."""
    # Data array is filled and we have to increase the size.
    new_data = scipy.empty(length * self.dim)
    if self._data is not None:
      new_data[:self.fill * self.dim] = self.data.ravel()
    self._data = new_data


class NumpyScalarsContainer(NumpyVectorsContainer):

  def __init__(self):
    """Create a NumpyScalarsContainer object."""
    super(NumpyScalarsContainer, self).__init__(1)

  def append(self, item):
    """Append a scalar to the container."""
    if self.freeRows() <= self.capacity:
      self.allocate(self.capacity)
    self._data[self.fill * self.dim:(self.fill + 1) * self.dim] = item
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

    start *= self.dim
    stop *= self.dim

    res = self.data[start:stop].ravel()
    res.shape = res.shape[0] / self.dim, self.dim
    return res

  def append(self, sequence):
    """Append a sequence to the container."""
    seqlength = len(sequence)
    sequence = scipy.asarray(sequence)
    while self.freeRows() < seqlength:
      self.allocate(self.capacity * 2)
    self._data[self.fill * self.dim:(self.fill + seqlength) * self.dim] = \
        sequence.ravel()
    self.sequenceStarts.append(self.fill)
    self.fill += seqlength


class ExternalVectorsContainer(Container):

  # Allow files up to this size in bytes.
  max_file_size = 128 * 1024 * 1024

  # Allow so many doubles per file.
  max_doubles_per_file = max_file_size / 8

  # Size of a double on this platform.
  doublesize = struct.calcsize('d')

  # Buffer size for files.
  buffersize = 32 * 1024 * 1024 

  def __init__(self, dim, datadir=".", keepFiles=False):
    self.dim = dim
    self.itemsPerFile = self.max_doubles_per_file / dim
    self.files = []

    # To avoid reopening the same file again and again, we keep the last used
    # file descriptor. This will hopefully be good enough for serial access and
    # appending.
    self._cur_filename = None
    self._cur_fd = None

    # If set to True, files will be kept over the lifetime of the object (And
    # possibly the process!)
    # TODO: establish serialization method instead.
    self.keepFiles = keepFiles
    self.datadir = datadir

  def __del__(self):
    if self._cur_fd is not None:
      self._cur_fd.close()
    if not self.keepFiles:
      for f in self.files:
        os.remove(f)

  def __getitem__(self, idx):
    self._cur_fd.flush()
    # Determine the file the vector resides in.
    fileidx, offset = divmod(idx, self.itemsPerFile)
    filename = self.files[fileidx]
    with open(filename) as fd:
      fd.seek(offset * self.doublesize)
      item = fread(fd, self.dim, 'd')
    return item

  def close(self):
    self._cur_fd.close()
    self._cur_fd = None
    self._cur_filename = None

  def fileForAppend(self, needed):
    filesize = os.path.getsize(self.files[-1]) if self.files else float('inf')
    if filesize + needed * self.doublesize >= self.max_file_size:
      # We need to create a new file if the previous file is approaching the
      # upper file limit.
      if self._cur_fd is not None:
        self._cur_fd.close()
      _fd = tempfile.NamedTemporaryFile('w', delete=False, dir=self.datadir)
      self._cur_fd = open(_fd.name, 'a', self.buffersize)
      self._cur_filename = _fd.name
      self.files.append(_fd.name)
    elif self._cur_filename != self.files[-1]:
      self._cur_fd.close()
      self._cur_fd = open(self.files[-1], 'a', self.buffersize)
      self._cur_filename = self.files[-1]
      #self._cur_fd.seek(0, 2)
    return self._cur_fd

  def append(self, item):
    item = scipy.asarray(item, dtype='float64')
    fd = self.fileForAppend(self.dim)
    fwrite(fd, self.dim, item)


class ExternalScalarsContainer(ExternalVectorsContainer):

  def __init__(self):
    super(ExternalScalarsContainer, self).__init__(1)

  def __getitem__(self, idx):
    return super(ExternalScalarsContainer, self).__getitem__(idx)[0]

  def append(self, item):
    super(ExternalScalarsContainer, self).append([item])


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
    self.fileToSequences = collections.defaultdict(list)

  def __getitem__(self, idx):
    fileidx = self.sequenceToFiles[idx]
    filename = self.files[fileidx]
    with open(filename) as fd:
      # Determine the indexes of the files that are in the same file before this
      # sequence.
      precedSeqs = [i for i in self.fileToSequences[fileidx] if i < idx]
      # Calculate the offset in the file.
      seq_offset = sum(self.sequenceToLengths[i] * self.dim for i in precedSeqs)
      # Seek to that offset...
      fd.seek(self.doublesize * seq_offset)
      # ... and return the corresponding sequence.
      res = fread(fd, self.sequenceToLengths[idx] * self.dim, 'd')
    res.shape = res.size / self.dim, self.dim

    return res

  def append(self, item):
    item = scipy.asarray(item, dtype='float64')
    fd = self.fileForAppend(item.size)
    fwrite(fd, item.size, item)

    fileidx = len(self.files) - 1
    self.sequenceToFiles.append(fileidx)
    self.sequenceToLengths.append(item.shape[0])
    self.fileToSequences[fileidx].append(len(self.sequenceToFiles) - 1)


class SharedVectorsContainer(ArrayContainer):

  @property
  def capacity(self):
    return len(self._data)

  def allocate(self, length):
    new_data = multiprocessing.Array('d', length)
    if self._data is not None:
      new_data[:len(self._data)] = self._data
    self._data = new_data

  def __getitem__(self, idx):
    item = self._data[idx * self.dim:(idx + 1) * self.dim]
    item = scipy.asarray(item)
    return item


class SharedScalarsContainer(SharedVectorsContainer):

  def __init__(self):
    super(SharedScalarsContainer, self).__init__(1)

  def __getitem__(self, idx):
    item = super(SharedVectorsContainer, self).__getitem__(idx)
    return item[0]

  def append(self, item):
    super(SharedVectorsContainer, self).append([item])


class SharedSequencesContainer(SharedVectorsContainer):

  def __init__(self, dim):
    """Create a SharedSequencesContainer object."""
    super(SharedSequencesContainer, self).__init__(dim)
    self.sequenceStarts = []

  def __getitem__(self, idx):
    start = self.sequenceStarts[idx]
    try: 
      stop = self.sequenceStarts[idx + 1]
    except IndexError:
      stop = self.fill

    start *= self.dim
    stop *= self.dim

    res = self._data[start:stop]
    res = scipy.asarray(res)
    res.shape = res.shape[0] / self.dim, self.dim
    return res

  def append(self, sequence):
    """Append a sequence to the container."""
    seqlength = len(sequence)
    sequence = scipy.asarray(sequence)
    while self.freeRows() < seqlength:
      self.allocate(self.capacity * 2)
    self._data[self.fill * self.dim:(self.fill + seqlength) * self.dim] = \
        sequence.ravel()
    self.sequenceStarts.append(self.fill)
    self.fill += seqlength


containerRegistry = {
  (Vectors, 'numpy'): NumpyVectorsContainer,
  (Scalars, 'numpy'): lambda _: NumpyScalarsContainer(),
  (Sequences, 'numpy'): NumpySequencesContainer,
  
  (Vectors, 'list'): lambda _: list(),
  (Scalars, 'list'): lambda _: list(),
  (Sequences, 'list'): lambda _: list(),
  
  (Vectors, 'external'): ExternalVectorsContainer,
  (Scalars, 'external'): lambda _: ExternalScalarsContainer(),
  (Sequences, 'external'): ExternalSequencesContainer,

  (Vectors, 'shared'): SharedVectorsContainer,
  (Scalars, 'shared'): lambda _: SharedScalarsContainer(),
  (Sequences, 'shared'): SharedSequencesContainer,

}





