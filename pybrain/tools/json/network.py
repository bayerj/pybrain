#! /usr/bin/env python2.5
# -*- coding: utf-8 -*-


from __future__ import with_statement


__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'


import itertools

try:
  import json
except ImportError, e:
  try:
    import simplejson as json
  except ImportError, e:
    raise ImportError("Need simplejson or python2.6")


import arac.pybrainbridge
import pybrain
from pybrain.structure.networks.recurrent import RecurrentNetworkComponent
from pybrain.structure.modules.module import Module
from pybrain.structure.connections.connection import Connection
from pybrain.structure.parametercontainer import ParameterContainer
from pybrain.structure import (Network, SigmoidLayer, TanhLayer, LinearLayer,
                               LSTMLayer, MDLSTMLayer, IdentityConnection,
                               FullConnection, LinearConnection, BiasUnit,
                               GateLayer, DoubleGateLayer, SwitchLayer,
                               MultiplicationLayer)
from pybrain.utilities import canonicClassString, multimethod


@multimethod(BiasUnit)
@multimethod(LinearLayer)
@multimethod(TanhLayer)
@multimethod(SigmoidLayer)
@multimethod(GateLayer)
@multimethod(DoubleGateLayer)
@multimethod(MultiplicationLayer)
@multimethod(SwitchLayer)
@multimethod(LSTMLayer)
@multimethod(MDLSTMLayer)
def dictRepresentation(obj):
  """Return the representation of a pybrain structure object as a python
  dict."""
  argdict = obj.argdict.copy()
  argdict['name'] = obj.name
  dct = {'class': canonicClassString(obj),
          'args': argdict, }
  if isinstance(obj, ParameterContainer):
    dct['parameters'] = obj.params.tolist()
  return dct


@multimethod(IdentityConnection)
@multimethod(LinearConnection)
@multimethod(FullConnection)
def dictRepresentation(obj):
  argdict = obj.argdict.copy()
  # These objects are not JSON serializable, thus they have to be excluded.
  del argdict['inmod']
  del argdict['outmod']
  argdict['name'] = obj.name
  dct = {'class': canonicClassString(obj),
         'args': argdict,
         'inmod': obj.inmod.name,
         'outmod': obj.outmod.name, }
  if isinstance(obj, ParameterContainer):
    dct['parameters'] = obj.params.tolist()
  return dct


def writeToFileObject(net, fileobject):
  """Return a network that was read from a file-like object in JSON format."""
  # Check if the network is a nested network. If so, throw an error, since
  # serializing those is net yet supported.
  if any(isinstance(i, Network) for i in net.modules):
    raise ValueError("JSON serialization for nested networks not supported.")
  # We will represent a network as a dictionary first which will then be
  # serialized by a call to the JSON library.
  mods = [dictRepresentation(i) for i in net.modules]
  cons = itertools.chain(*net.connections.values())
  cons = [dictRepresentation(i) for i in cons]
  inmods = [i.name for i in net.inmodules]
  outmods = [i.name for i in net.outmodules]
  
  netdict = {
    'name': net.name,
    'class': canonicClassString(net),
    'inmodules': inmods,
    'outmodules': outmods,
    'modules': mods,
    'connections': cons,
  }
  if isinstance(net, RecurrentNetworkComponent):
    netdict['recurrentcons'] = [i.name for i in net.recurrentConns]
    reccons = [dictRepresentation(i) for i in net.recurrentConns]
    netdict['connections'] += reccons

  # Then serialize connections and use module ids as inmodule/outmodule specs.
  json.dump(netdict, fileobject, indent=2)


def readFromFileObject(fileobject):
  """Write a network to a file-like object in JSON format."""
  dct = json.load(fileobject)

  # First recover modules.
  mods = []
  for i in dct['modules']:
    args = dict((str(k), v) for k, v in i['args'].items())
    mod = eval(i['class'])(**args) 
    if 'parameters' in i:
      mod.params[:] = i['parameters']
    mods.append(mod)
  mods = dict((str(i.name), i) for i in mods)

  print mods

  # Then recover connections.
  cons = []
  for i in dct['connections']:
    # Replace names of modules by their new instances.
    i['args']['inmod'] = mods[str(i['inmod'])]
    i['args']['outmod'] = mods[str(i['outmod'])]
    args = dict((str(k), v) for k, v in i['args'].items())
    con = eval(i['class'])(**args)
    if 'parameters' in i:
      con.params[:] = i['parameters']
    cons.append(con)

  net = eval(dct['class'])(name=dct['name'])
  for name, mod in mods.items():
    net.addModule(mod)
    if mod.name in dct['inmodules']:
      net.addInputModule(mod)
    if mod.name in dct['outmodules']:
      net.addOutputModule(mod)

  try:
    recurrentcons = set(dct['recurrentcons'])
  except KeyError:
    recurrentcons = set()
  for con in cons:
    if con.name in recurrentcons:
      net.addRecurrentConnection(con)
    else:
      net.addConnection(con)

  net.sortModules()
  return net


def writeToFile(net, filename):
  """Write a network to a file in JSON format."""
  with open(filename, 'w+') as fp:
    return writeToFileObject(net, fp)


def readFromFile(filename):
  """Return a network that was read from a file in JSON format."""
  with open(filename, 'r') as fp:
    return readFromFileObject(fp)
