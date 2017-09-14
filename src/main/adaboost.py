from sklearn.cross_validation import ShuffleSplit, StratifiedShuffleSplit
from  model import Model
import numpy as np
from numpy.core.umath_tests import inner1d
from collections import defaultdict as dd
import sys
from numpy.random import sample

class AdaBoost(Model):
  
  def __init__(self, modelList):
    super(AdaBoost, self).__init__()
    self.modelList = modelList
    
  def getTrainTestSplits(self, y, n, fractionTest):
    try:
      return StratifiedShuffleSplit(y, n_iter=n, test_size=fractionTest)
    except ValueError:
      return ShuffleSplit(y.shape[0], n_iter=n, test_size=fractionTest)
  
  def crossValidate(self, x, y, nrTrainTestSplits=1, fractionTest = 0.25, verbose=False):
    
    avgF1score = 0
    avgAccscore = 0
    avgLogLoss = 0
    
    splits = self.getTrainTestSplits(y, nrTrainTestSplits, fractionTest)
    splitIndex = 0
    
    for train_index, test_index in splits:
      self.model = []
      # get train-test split
      xTrain, xTest = x[train_index], x[test_index]
      yTrain, yTest = y[train_index], y[test_index]
      
      self.classToIndex = Model.mapClasses(yTrain)
      
      sampleWeights = np.ones(yTrain.shape)
      
      if verbose:
        #print('sampleWeights: {}, sum:{}, abs sum:{}'.format(sampleWeights[:20], np.sum(sampleWeights), np.sum(np.abs(sampleWeights))))
        print('split {:d} of {:d}'.format((splitIndex+1), len(splits)))

      self.learn(xTrain, yTrain, xHeldout=xTest, yHeldout=yTest, weights=sampleWeights, verbose=verbose, classToIndex=self.classToIndex)
      
      (predLabelsTest, predProbTest) = self.predict(xTest)
      (acc, F1, logLoss) = self.__getPerformanceScores(yTest, predLabelsTest, predProbTest, np.ones(yTest.shape[0]), self.classToIndex)
      # check if we need to put this model as the best performing model so far
      if verbose:
        print('Final performance in split {:d} on held-out fold: {}'.format((splitIndex+1), self.__getPerformanceString(acc, F1, logLoss)))
      avgF1score += F1
      avgAccscore += acc
      avgLogLoss += logLoss
      splitIndex += 1
    
    avgF1score /= float(splitIndex)
    avgAccscore /= float(splitIndex)
    avgLogLoss /= float(splitIndex)
    
    if verbose:
      print('Avg. scores over splits: {}'.format(self.__getPerformanceString(avgAccscore, avgF1score, avgLogLoss)))
    # preserve best performing model (over train-test splits) as the final model
    self.model = None
    return (avgAccscore, avgF1score, avgLogLoss)
  
  
  def learn(self, x, y, xHeldout=[], yHeldout=[], weights=[], verbose=False, classToIndex=None):
    '''
      updates self.model
    '''  
    self.classToIndex = Model.mapClasses(y) if classToIndex == None else classToIndex
    sampleWeights = np.ones(y.shape) if len(weights) == 0 else weights
    
    nrModels = len(self.modelList)
    # train each submodel
    for i in range(nrModels):
      
      if verbose:
        print('model {:d} of {:d}'.format((i+1), nrModels))
        
      (newSampleWeights, acc, F1, logLoss) = self.__singleIteration(self.modelList[i], sampleWeights, x, y, self.classToIndex)
      
      if verbose:
        print('Iteration {:d}, model scores (using sample weights): {}'.format((i+1), self.__getPerformanceString(acc, F1, logLoss)))
        #print('sampleWeights: {}, sum:{}, abs sum:{}'.format(sampleWeights[:20], np.sum(sampleWeights), np.sum(np.abs(sampleWeights))))
        
        self.model = self.modelList[:i+1]
        (predLabelsFinal_i, predProbFinal_i) = self.predict(x)
        (acc, F1, logLoss) = self.__getPerformanceScores(y, predLabelsFinal_i, predProbFinal_i, np.ones(y.shape[0]), self.classToIndex)
        print('Iteration {:d}, adaboost scores combined model (uniform weights, unnormalized decision function): {}'.
              format((i+1), self.__getPerformanceString(acc, F1, logLoss)))
        
        if verbose and min(len(xHeldout), len(yHeldout)) > 0:
          (predLabelsHeldout, predProbHeldout) = self.predict(xHeldout)
          (acc, F1, logLoss) = self.__getPerformanceScores(yHeldout, predLabelsHeldout, predProbHeldout, np.ones(yHeldout.shape[0]), self.classToIndex)
          print('Iteration {:d}, adaboost performance on held-out set: {}'.format((i+1), self.__getPerformanceString(acc, F1, logLoss)))
     
      if (i < (nrModels -1)):
        sampleWeights = newSampleWeights
      
    self.model = self.modelList
    predLabelsFinal, predProbFinal = self.predict(x)
    if verbose:
      (acc,F1, logLossAverage) = self.__getPerformanceScores(y, predLabelsFinal, predProbFinal, np.ones(y.shape[0]), self.classToIndex)
      print('Final performance train set (uniform weights): {}'.format(self.__getPerformanceString(acc, F1, logLossAverage)))
    return (predLabelsFinal, predProbFinal, self.classToIndex)
  
  @staticmethod
  def __getPerformanceString(acc, F1, logLoss):
    return ('Accuracy: {0:2.4f}, F1: {1:2.4f}, Log loss = {2:2.4f}'.format(acc, F1, logLoss)).rstrip()
  
  def __getPerformanceScores(self, y, predLabels, predProb, sampleWeights, classToIndex):
    (acc,F1) = Model.getCommonMetricScores(y, predLabels, sampleWeights=sampleWeights)
    (logLossAverage, _logLossPerSample) = Model.getLogLoss(y, predProb, classToIndex, sampleWeights=sampleWeights)
    return (acc, F1, logLossAverage)
    
  def __singleIteration(self, model, sampleWeights, x, y, classToIndex):
      (predLabels, predProb, _) = model.learn(x, y, weights=sampleWeights, classToIndex=classToIndex, verbose=True)
      
      (acc,F1, logLossAverage) = self.__getPerformanceScores(y, predLabels, predProb, sampleWeights, classToIndex)
      
      predProb[predProb < np.finfo(predProb.dtype).eps] = np.finfo(predProb.dtype).eps
      predProbLog = np.log(predProb)
      
      sampleWeights = self.__updateSampleWeights__(predProbLog, classToIndex, y, sampleWeights)
      
      return (sampleWeights, acc, F1, logLossAverage)
      
      
  def __updateSampleWeights__(self, predProbLog, classToIndex, yTrain, sampleWeights):
    
    nrClasses = len(classToIndex)

    yArray = np.repeat([-1/float(nrClasses-1)], nrClasses)
    yMatrix = np.repeat([yArray], predProbLog.shape[0], axis=0)
    #print('yMatrix.shape: {}, predProbLog.shape:{}, yTrain.shape: {}'.format(yMatrix.shape, predProbLog.shape, yTrain.shape))
    indices = list(map(lambda x:classToIndex[x], yTrain))
    yMatrix[list(range(0, yTrain.shape[0])),indices]=1
   
    wMult = np.exp(
                    (-(nrClasses-1)/float(nrClasses)) * 
                      inner1d(yMatrix, predProbLog)
                  )
    newWeights=  sampleWeights * wMult
    newWeightsNormalized = newWeights / np.sum(newWeights) * yTrain.shape[0]
    return newWeightsNormalized
#    
  def __getSammeRprobabilities(self, predProbLog):
    
    nrClasses = len(self.classToIndex)
    sumOverClasses = np.sum(predProbLog,axis=1)
    #print(' sumOverClasses: {}',sumOverClasses)
    h = (nrClasses-1) * (
         predProbLog  - 
            (1/float(nrClasses)) * 
              #predProbLog.sum(axis=1)[:, np.newaxis]
              np.repeat(np.transpose([sumOverClasses]),predProbLog.shape[1],axis=1)
        )
    return h
  
  # predict a new input set, based on the learned model 
  # by taking into account the predicted labels of all submodels, as well as the weight per submodel
  def predict(self, newX):
    if self.model == None or len(self.model)==0:
      raise RuntimeError('no model learnt yet')
    modelList = self.model
    
    nrClasses = len(self.classToIndex)
    h_sum = np.zeros((newX.shape[0], nrClasses))
    i = 0
    for i in range(len(modelList)):
      #print(i)
      predLabels, predProb = modelList[i].predict(newX)
      predProb[predProb < np.finfo(predProb.dtype).eps] = np.finfo(predProb.dtype).eps
      predProbLog = np.log(predProb)
      h_i = self.__getSammeRprobabilities(predProbLog)
      h_sum += h_i
      #print(np.sum(h_sum))
    
    predProb = h_sum / len(modelList)
    
    
    # "np.exp((1./(nrClasses-1)) * predProb)" undoes the "(nrClasses-1)*np.log(predProb)" part, 
    # makes the resulting probabilities of boosting with 1 model (1 adaboost iteration) the same as when 
    # just using that model.
    # used in scikit when estimating probabilities.
    # also increases the entropy of the probabilities higher and the log loss, even when accuracy increases - 
    # hence not used.
    #predProb = np.exp((1. / (nrClasses - 1)) * predProb) 
    
   
    predProb = self.__normalize(predProb)
    
    bestClassIndices = np.argmax(h_sum, axis=1)
    predLabels = Model.unmapClasses(bestClassIndices, self.classToIndex)
    
    return (predLabels, predProb)

  def __normalize(self, predProb):
    normalizer = predProb.sum(axis=1)[:, np.newaxis]
    normalizer[normalizer == 0.0] = 1.0
    predProb /= normalizer
    return predProb
  
if __name__=="__main__":
  s = AdaBoost.getPerformanceString(.44434, .84353, 2.34545)
  print('hi {}'.format(s))
  print('hi {}'.format(s))
        
      
        
      