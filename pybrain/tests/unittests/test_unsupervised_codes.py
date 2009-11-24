__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'


import unittest

from pybrain.tests import runModuleTestSuite
from pybrain.unsupervised.codes.pca import PcaCode
from pybrain.datasets.unsupervised import UnsupervisedDataSet


class TestPcaCode(unittest.TestCase):

  def setUp(self):
    data = [[2.5, 2.4], 
            [0.5, 0.7], 
            [2.2, 2.9], 
            [1.9, 2.2], 
            [3.1, 3.0], 
            [2.3, 2.7], 
            [2.0, 1.6], 
            [1.0, 1.1], 
            [1.5, 1.6], 
            [1.1, 0.9]]

    self.dataset = UnsupervisedDataSet(2)
    for row in data:
      self.dataset.appendLinked(row)

  def testSimple(self):
    code = PcaCode(2, 1)
    code.learn(self.dataset)
  
    self.assertEqual(code.encoding[0, 0], -0.67787339852801165)
    self.assertEqual(code.encoding[0, 1], -0.73517865554440798)

    input = 2.5, 2.4
    coded = code.encode((2.5, 2.4))
    decoded = code.decode(coded)

    self.assertEqual(decoded[0], 2.3712589640000026)
    self.assertEqual(decoded[1], 2.5187060083221686)
  


if __name__ == "__main__":
    runModuleTestSuite(__import__('__main__'))

