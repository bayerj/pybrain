__author__ = 'Daan Wierstra and Tom Schaul'


import sys
import itertools
import Queue 
from random import shuffle

import scipy
from scipy import dot, argmax


try: 
  import multiprocessing as mp# For python 2.6
except ImportError:
  try:
    import pyprocessing as mp# For python < 2.5
  except ImportError:
    # Fail silently.
    pass


from trainer import Trainer
from pybrain.utilities import fListToString 
from pybrain.auxiliary import GradientDescent


class BackpropTrainer(Trainer):
    """Trainer that trains the parameters of a module according to a 
    supervised dataset (potentially sequential) by backpropagating the errors
    (through time)."""
        
    def __init__(self, module, dataset=None, learningrate=0.01, lrdecay=1.0,
                 momentum=0., verbose=False, batchlearning=False,
                 weightdecay=0.):
        """Create a BackpropTrainer to train the specified `module` on the 
        specified `dataset`.
        
        The learning rate gives the ratio of which parameters are changed into 
        the direction of the gradient. The learning rate decreases by `lrdecay`, 
        which is used to to multiply the learning rate after each training 
        step. The parameters are also adjusted with respect to `momentum`, which 
        is the ratio by which the gradient of the last timestep is used.
        
        If `batchlearning` is set, the parameters are updated only at the end of
        each epoch. Default is False.
        
        `weightdecay` corresponds to the weightdecay rate, where 0 is no weight
        decay at all.
        """
        Trainer.__init__(self, module)
        self.ds = dataset
        self.verbose = verbose
        self.batchlearning = batchlearning
        self.weightdecay = weightdecay
        self.epoch = 0
        self.totalepochs = 0
        # set up gradient descender
        self.descent = GradientDescent()
        self.descent.alpha = learningrate
        self.descent.momentum = momentum
        self.descent.alphadecay = lrdecay
        self.descent.init(module.params)
        
    def shuffleInputs(self):
        """Generator function that yields dataset entries in random order."""
        indices = range(len(self.ds))
        shuffle(indices)
        for i in indices:
          yield self.ds[i]

    def train(self):
        """Train the associated module for one epoch."""
        assert len(self.ds) > 0, "Dataset cannot be empty."
        self.module.resetDerivatives()
        errors = 0        
        ponderation = 0.

        for seq in self.shuffleInputs():
            e, p = self._calcDerivs(seq)
            errors += e
            ponderation += p
            if not self.batchlearning:
                gradient = self.module.derivs - self.weightdecay * self.module.params
                new = self.descent(gradient, errors)
                if new is not None:
                    self.module.params[:] = new
                self.module.resetDerivatives()

        if self.verbose:
            print "Total error:", errors / ponderation
        if self.batchlearning:
          self.module.params[:] = self.descent(self.module.derivs)
        self.epoch += 1
        self.totalepochs += 1
        return errors / ponderation
    
    def _calcDerivs(self, seq):
        """Calculate error function and backpropagate output errors to yield 
        the gradient."""
        seq = zip(*seq)     # TODO: this should be done better, I guess.
        self.module.reset()     
        for sample in seq:
            self.module.activate(sample[0])
        error = 0
        ponderation = 0.
        for offset, sample in reversed(list(enumerate(seq))):
            # need to make a distinction here between datasets containing
            # importance, and others
            target = sample[1]
            outerr = target - self.module.outputbuffer[offset]
            if len(sample) > 2:
                importance = sample[2]
                error += 0.5 * dot(importance, outerr ** 2)
                ponderation += sum(importance)
                self.module.backActivate(outerr * importance)                
            else:
                error += 0.5 * sum(outerr ** 2)
                ponderation += len(target)
                # FIXME: the next line keeps arac from producing NaNs. I don't
                # know why that is, but somehow the __str__ method of the 
                # ndarray class fixes something,
                str(outerr)
                self.module.backActivate(outerr)
            
        return error, ponderation
            
    def _checkGradient(self, dataset=None, silent=False):
        """Numeric check of the computed gradient for debugging purposes."""
        if dataset:
            self.setData(dataset)
        res = []
        for seq in self.ds._provideSequences():
            self.module.resetDerivatives()
            self._calcDerivs(seq)
            e = 1e-6    
            analyticalDerivs = self.module.derivs.copy()
            numericalDerivs = []
            for p in range(self.module.paramdim):
                storedoldval = self.module.params[p]
                self.module.params[p] += e
                righterror, dummy = self._calcDerivs(seq)
                self.module.params[p] -= 2 * e
                lefterror, dummy = self._calcDerivs(seq)
                approxderiv = (righterror - lefterror) / (2 * e)
                self.module.params[p] = storedoldval
                numericalDerivs.append(approxderiv)
            r = zip(analyticalDerivs, numericalDerivs)
            res.append(r)
            if not silent:
                print r
        return res
    
    def testOnData(self, dataset=None, verbose=False):
        """Compute the MSE of the module performance on the given dataset.

        If no dataset is supplied, the one passed upon Trainer initialization is
        used."""
        if dataset == None:
            dataset = self.ds
        dataset.reset()
        if verbose:
            print '\nTesting on data:'
        errors = []
        importances = []
        ponderatedErrors = []
        for seq in dataset._provideSequences():
            self.module.reset()
            e, i = dataset._evaluateSequence(self.module.activate, seq, verbose)
            importances.append(i)
            errors.append(e)
            ponderatedErrors.append(e / i)
        if verbose:
            print 'All errors:', ponderatedErrors
        assert sum(importances) > 0
        avgErr = sum(errors) / sum(importances)
        if verbose:
            print 'Average error:', avgErr
            print ('Max error:', max(ponderatedErrors), 'Median error:',
                   sorted(ponderatedErrors)[len(errors) / 2])
        return avgErr
                
    def testOnClassData(self, dataset=None, verbose=False,
                        return_targets=False):
        """Return winner-takes-all classification output on a given dataset. 
        
        If no dataset is given, the dataset passed during Trainer 
        initialization is used. If return_targets is set, also return 
        corresponding target classes.
        """
        if dataset == None:
            dataset = self.ds
        dataset.reset()
        out = []
        targ = []
        for seq in dataset._provideSequences():
            self.module.reset()
            for input, target in seq:
                res = self.module.activate(input)
                out.append(argmax(res))
                targ.append(argmax(target))
        if return_targets:
            return out, targ
        else:
            return out
        
    def trainUntilConvergence(self, dataset=None, maxEpochs=None, verbose=None,
                              continueEpochs=10, validationProportion=0.25):
        """Train the module on the dataset until it converges.
        
        Return the module with the parameters that gave the minimal validation 
        error. 
        
        If no dataset is given, the dataset passed during Trainer 
        initialization is used. validationProportion is the ratio of the dataset
        that is used for the validation dataset.
        
        If maxEpochs is given, at most that many epochs
        are trained. Each time validation error hits a minimum, try for 
        continueEpochs epochs to find a better one."""
        epochs = 0
        if dataset == None:
            dataset = self.ds
        if verbose == None:
            verbose = self.verbose
        # Split the dataset randomly: validationProportion of the samples for 
        # validation.
        trainingData, validationData = (
            dataset.splitWithProportion(1 - validationProportion))
        if not (len(trainingData) > 0 and len(validationData)):
            raise ValueError("Provided dataset too small to be split into training " + 
                             "and validation sets with proportion " + str(validationProportion))
        self.ds = trainingData
        bestweights = self.module.params.copy()
        bestverr = self.testOnData(validationData)
        trainingErrors = []
        validationErrors = [bestverr]
        while True:
            trainingErrors.append(self.train())
            validationErrors.append(self.testOnData(validationData))
            if epochs == 0 or validationErrors[-1] < bestverr:
                # one update is always done
                bestverr = validationErrors[-1]
                bestweights = self.module.params.copy()
            
            if maxEpochs != None and epochs >= maxEpochs:
                self.module.params[:] = bestweights
                break
            epochs += 1
            
            if len(validationErrors) >= continueEpochs * 2:
                # have the validation errors started going up again?
                # compare the average of the last few to the previous few
                old = validationErrors[-continueEpochs * 2:-continueEpochs]
                new = validationErrors[-continueEpochs:]
                if min(new) > max(old):
                    self.module.params[:] = bestweights
                    break
        trainingErrors.append(self.testOnData(trainingData))
        self.ds = dataset
        if verbose:
            print 'train-errors:', fListToString(trainingErrors, 6)
            print 'valid-errors:', fListToString(validationErrors, 6)
        return trainingErrors, validationErrors


class DerivWorker(mp.Process):

  def __init__(self, module, dataset, queue, conn):
    self.module = module.convertToFastNetwork()
    self.dataset = dataset
    self.queue = queue
    self.conn = conn
    super(DerivWorker, self).__init__()

  def run(self):
    while True:
      # Busy-wait until new parameters have arrived.
      while not self.conn.poll(0.1):
        pass
      packed = self.conn.recv()

      # Break main loop if 'finished' is being sent. 
      if packed == 'finished':
        break 

      # Otherwise trust other side that it's an array with new parameters.
      self.module.params[:] = packed 

      # Work.
      error = []
      ponderation = []
      derivs = []
      while True:
        try:
          jobs = self.queue.get(True, 0.1)
        except Queue.Empty, e:
          break
        for i in jobs:
          e, p, d = calcDeriv(self.module, self.dataset[i]) 
          error.append(e)
          ponderation.append(p)
          derivs.append(d)
        self.queue.task_done()
      self.conn.send((error, ponderation, derivs))


def calcDeriv(module, seq):
  # TODO: Copypasted from above. Should be factored out. This is actually a
  # dangerous piece of code, since it seems very similar to the above -
  # however, there are some differences: eg derivs are cleared before new
  # calculations, and also returned.
  seq = zip(*seq)
  module.reset()     

  for sample in seq:
      module.activate(sample[0])
  error = 0
  ponderation = 0.
  for offset, sample in reversed(list(enumerate(seq))):
      # need to make a distinction here between datasets containing
      # importance, and others
      target = sample[1]
      outerr = target - module.outputbuffer[offset]
      if len(sample) > 2:
          importance = sample[2]
          error += 0.5 * dot(importance, outerr ** 2)
          ponderation += sum(importance)
          module.backActivate(outerr * importance)                
      else:
          error += 0.5 * sum(outerr ** 2)
          ponderation += len(target)
          # FIXME: the next line keeps arac from producing NaNs. I don't
          # know why that is, but somehow the __str__ method of the 
          # ndarray class fixes something,
          str(outerr)
          module.backActivate(outerr)
      
  return error, ponderation, module.derivs.copy()
  

class ParallelBackpropTrainer(BackpropTrainer):

    def __init__(self, network, dataset, numProcesses=None):
      super(ParallelBackpropTrainer, self).__init__(network, dataset)
      self.numProcesses = numProcesses if numProcesses else mp.cpu_count()
      self._initPool()

    def _initPool(self):
      """Initialize working state for other processes."""
      conns = [mp.Pipe() for _ in range(self.numProcesses)]
      self.conns = [j for i, j in conns]
      self.queue = mp.JoinableQueue()
      self.pool = [DerivWorker(self.module, self.ds, self.queue, c) 
                   for c, _ in conns]
      for p in self.pool:
        p.start()

    def distList(self, lst, chunksize):
      chunks, lastchunk = divmod(len(lst), chunksize)
      if lastchunk:
        chunks += 1
      for i in xrange(chunks):
        yield lst[i * chunksize:(i + 1) * chunksize]
        
    def trainBatch(self, idxs):
      # Distribute jobs. Do this first, to make sure the queue is filled when
      # the workers start working. (They first wait for parameters.)

      jobs = list(self.distList(idxs, 1000))

      for i in jobs:
        self.queue.put(i)

      # Send new params
      for conn in self.conns:
        conn.send(self.module.params.copy())
      
      self.queue.join()
      
      # Retrieve updates.
      sys.stdout.flush()

      results = []
      for conn in self.conns:
        results.append(conn.recv())

      return results

    def train(self):
      idxs = range(len(self.ds))
      shuffle(idxs)
      batchsize = 10000
      nBatches, more = divmod(len(idxs), batchsize)
      if more:
        nBatches += 1
      batches = [idxs[i * batchsize:(i + 1) * batchsize] 
                 for i in xrange(nBatches)]

      error = 0
      ponderation = 0
      for batch in batches:
        results = self.trainBatch(batch)
        derivs = scipy.zeros(self.module.params.shape)
      
        # TODO: derivs are just added here, there should be specific
        # functionality.
        for e, p, d in results:
          error += sum(e)
          ponderation += sum(p)
          derivs += sum(d)
        self.module.params[:] = self.descent(derivs)

      return error / ponderation

    def __del__(self):
      for conn in self.conns:
        conn.send('finished')
