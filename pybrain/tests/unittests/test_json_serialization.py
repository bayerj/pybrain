#! /usr/bin/env python2.5
# -*- coding: utf-8 -*-


__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'


import tempfile
import unittest

from pybrain.structure import *
from pybrain.tools.shortcuts import buildNetwork
from pybrain.tests import runModuleTestSuite
from pybrain.tests.helpers import netCompare
from pybrain.tools.json.network import writeToFileObject, readFromFileObject



class JsonSerializationTest(unittest.TestCase):
  
  def jsonInvariance(self, net):
    fp = tempfile.TemporaryFile()
    writeToFileObject(net, fp)
    fp.seek(0)
    othernet = readFromFileObject(fp)
    return netCompare(net, othernet)

  def testFeedForwardSigmoid(self):
    n = buildNetwork(2, 3, 1, hiddenclass=SigmoidLayer)
    self.assert_(self.jsonInvariance(n))

  def testRecurrent(self):
    n = buildNetwork(2, 3, 1, hiddenclass=SigmoidLayer, recurrent=True)
    self.assert_(self.jsonInvariance(n))

  def testArac(self):
    n = buildNetwork(2, 3, 1, hiddenclass=SigmoidLayer,
                     fast=True)
    self.assert_(self.jsonInvariance(n))


if __name__ == "__main__":
    runModuleTestSuite(__import__('__main__'))

