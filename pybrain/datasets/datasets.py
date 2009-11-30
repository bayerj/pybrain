#!/usr/bin/env python2.5
# -*- coding: utf-8 -*-


__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'


from pybrain.datasets.base import DataSet, Vectors, Scalars, Sequences


class RegressionDataSet(DataSet):

  inputs = Vectors()
  targets = Vectors()


class SequenceRegressionDataSet(DataSet):

  inputs = Sequences()
  targets = Sequences()
