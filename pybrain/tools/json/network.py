#! /usr/bin/env python2.5
# -*- coding: utf-8 -*-

__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'



def writeToFileObject(net, fileobject):
  pass


def readFromFileObject(fileobject):
  pass


def writeToFile(net, filename):
  with open(filename, 'w+') as fp:
    return writeToFileLike(net, fp)


def readFromFile(filename):
  with open(filename, 'r') as fp:
    return readFromFileObject(fp)
