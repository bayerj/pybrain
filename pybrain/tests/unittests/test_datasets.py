#!/usr/bin/env python2.5
# -*- coding: utf-8 -*-


__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'


import unittest

from pybrain.tests import runModuleTestSuite

from pybrain.datasets import Vectors, Scalars, Sequences, DataSet


class TestListDataSet(unittest.TestCase):

  containertype = 'list'

  def setUp(self):
    class SomeDataSet(DataSet):
      inputs = Sequences()
      targets = Vectors()

    self.SomeDataSet = SomeDataSet

  def testFieldOrdering(self):
    self.assertEqual([k for k, v in self.SomeDataSet._fieldspecs], 
                     ['inputs', 'targets'])

  def testFromIter(self):
    inputs = [[(1, 2), (3, 4), (4, 5)],
             [(0, 0), (1, 1)]]
    targets = [(2, 0), (3, 4)]
    ds = self.SomeDataSet.fromIter([inputs, targets],
                                   containertype=self.containertype)

    self.assertEqual(ds.inputs[0][0][0], 1)
    self.assertEqual(ds.inputs[0][0][1], 2)
    self.assertEqual(ds.inputs[1][1][1], 1)
    self.assertEqual(ds.targets[0][0], 2)

  def testFromSizes(self):
    ds = self.SomeDataSet([2, 1], containertype=self.containertype)
    ds.inputs.append([(1, 2), (3, 4), (4, 5)])
    ds.targets.append([3])

    ds.inputs.append([(0, 0), (1, 1)])
    ds.targets.append([2])

    self.assertEqual(ds.inputs[0][0][0], 1)
    self.assertEqual(ds.inputs[0][0][1], 2)
    self.assertEqual(ds.inputs[1][1][1], 1)
    self.assertEqual(ds.targets[0][0], 3)
    self.assertEqual(ds.targets[1][0], 2)


class TestNumpyContainerType(TestListDataSet):

  containertype = 'numpy'


class TestExternalContainerType(TestListDataSet):

  containertype = 'external'


if __name__ == "__main__":
    runModuleTestSuite(__import__('__main__'))

