#! /usr/bin/env python2.5
# -*- coding: utf-8 -*-

__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'


from __future__ import with_statement


def writeToFileObject(net, fileobject):
  """Return a network that was read from a file-like object in JSON format."""
  # First serialzie all the modules of the network in a recursive fashion.
  # Then serialize connections and use module ids as inmodule/outmodule specs.
  pass


def readFromFileObject(fileobject):
  """Write a network to a file-like object in JSON format."""
  pass


def writeToFile(net, filename):
  """Write a network to a file in JSON format."""
  with open(filename, 'w+') as fp:
    return writeToFileLike(net, fp)


def readFromFile(filename):
  """Return a network that was read from a file in JSON format."""
  with open(filename, 'r') as fp:
    return readFromFileObject(fp)
