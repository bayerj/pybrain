#!/usr/bin/env python2.5


"""Script that plots the jacobian of a recurrent neural network."""


__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'


import optparse

import pylab

from pybrain.datasets import SequentialDataSet
from pybrain.tools.xml import NetworkReader


def make_optparser():
  parser = optparse.OptionParser()
  parser.add_option('-d', '--datafile', dest='datafile', type='string',
                    help='location of dataset file')
  parser.add_option('-n', '--networkfile', dest='networkfile', type='string',
                    help='location of network file')
  parser.add_option('-s', '--sequenceindex', dest='sequenceindex', type='int',
                    help='index of the sequence to use')
  parser.add_option('-i', '--input', dest='inputindex', type='int',
                    help='jacobian is calculated with respect to this input')
  parser.add_option('-o', '--output', dest='outputindex', type='int',
                    help='jacobian of this output is calculated')
  parser.add_option('-t', '--timestep', dest='timestep', type='int',
                    help='''the jacobian of the output at this timestep is
                    calculated.''')
  return parser


def calcJacobian(network, sequence, inputindex, outputindex):
  outputs = []
  for i, item in enumerate(seq):
    output = network.activate(item)
    newoutput[:] = [0] * dataset.outdim
    if i == options.timestep:
      newoutput[outputindex] = output[outputindex]
    outputs.append(newoutput)
  outputs.reverse()
  jacobian = []
  for o in outputs:
    deriv = network.backActivate(o)
    jacobian.append(deriv)
  return jacobian


def main():
  options, args = make_optparser().parse_args()
  network = NetworkReader.readFrom(options.networkfile)
  dataset = SequentialDataSet.loadFromFile(options.datafile)
  seq = dataset.getSequence(options.sequenceindex)
  jacobian = calcJacobian(network, dataset, seq, options.inputindex,
                          options.outputindex)

  pylab.plot(jacobian)
  pylab.show()


if __name__ == '__main__':
  main()




