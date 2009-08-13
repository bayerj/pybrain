# -*- coding: utf-8 -*-

__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'


import optparse
import pybrain.datasets.dsjson as dsjson
from pybrain.tools.xml import NetworkWriter, NetworkReader
from pybrain.supervised.trainers import BackpropTrainer


def make_optparser():
  parser = optparse.OptionParser()
  parser.add_option('-m', '--modulefile', 
                    help='file to load module from',
                    type='string',
                    dest='modulefile')
  parser.add_option('-d', '--datafile',
                    help='file to load dataset from',
                    type='string',
                    dest='datafile')
  parser.add_option('-l', '--learningrate',
                    help='backprop learningrate',
                    type='float',
                    default=0.05,
                    dest='learningrate')
  parser.add_option('-M', '--momentum',
                    help='backprop momentum',
                    type='float',
                    default=0.0,
                    dest='momentum')
  parser.add_option('-e', '--epochs',
                    type='int',
                    default=100,
                    help='epochs to train',)
                    
  return parser



def main():
  options, args = make_optparser().parse_args()

  print "Loading dataset...", 
  dataset = dsjson.unserializeSequential(options.datafile)
  print "ok"

  print "Loading network...",
  network = NetworkReader.readFrom(options.modulefile)
  print "ok"

  trainer = BackpropTrainer(network, dataset, 
                            learningrate=options.learningrate,
                            momentum=options.momentum)

  for _ in xrange(options.epochs):
    print trainer.train()


if __name__ == '__main__':
  main()
