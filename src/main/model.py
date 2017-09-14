import os
import abc
import sys
from abc import abstractmethod
import xgboost as xgb
from sklearn import metrics
import numpy as np
import tensorflow as tf
from multiprocessing import cpu_count
from sklearn.cross_validation import ShuffleSplit, StratifiedShuffleSplit

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
tf.logging.set_verbosity(tf.logging.WARN)

# only use newer abstract base class syntax when we're using Python >= 3.4
if sys.version_info >= (3, 4):
  ABC = abc.ABC
else:
  ABC = abc.ABCMeta(str('ABC'), (), {})

class Model(ABC):
  
  def __init__(self):
    self.model = None
  
  @abstractmethod
  def learn(self, x, y, weights=[], verbose=False, classToIndex=None):
    '''
    x: NxM array where N is number of samples and M the number of features
    y: 1D array of length N with the true class label per sample
    weights (optional): list of length N that indicates how much each sample should be
                        taken into account (relatively) by the learning algorithm. Default: all equal.
    verbose (optional): if True, print scores after learning. Default: False.
    classToIndex (optional): dictionary mapping each class label in y to its index in the
                              probability matrix
    returns: tuple of size 3 with:
              predicted labels: 1D array of length N with predicted class per sample
              predicted probabilities: NxD array where D is the number of classes
              class to index dictionary: dictionary from class label to column index in 
                the above predicted probabilities matrix     
    '''
    pass
  
  @abstractmethod
  def predict(self, newX):
    '''
    newX: NxM array where N is number of samples and M the number of features
    
    returns: tuple of size 2 with:
      predicted labels: 1D array of length N with predicted class per sample
      predicted probabilities: NxD array where D is the number of classes
    '''
    pass
  
  @staticmethod
  def getCommonMetricScores(yTrue, yPredLabels, sampleWeights=None):
    accScore = metrics.accuracy_score(yTrue, yPredLabels, sample_weight=sampleWeights)
    F1score = metrics.f1_score(yTrue, yPredLabels, average='weighted', sample_weight = sampleWeights) 
    return (accScore, F1score)
  
  @staticmethod
  def getLogLoss(yTrue, predProb, classToIndex, sampleWeights=[]):
    ''' 
    yTrue: 1D array with true class label per samples
    predProb: NxD array where N is nr. samples and D number labels. 
    classToIndex: a dictionary mapping each class label to its column index in predProb
    returns: tuple with the log loss average over sample (float) and 1D array with log loss per sample
    '''
    sampleWeights = np.ones(yTrue.shape[0]) if len(sampleWeights) == 0 else sampleWeights

    nrSamples = yTrue.shape[0]
    logLossPerSample = np.zeros(nrSamples)
    actualProb = np.zeros((yTrue.shape[0], len(classToIndex)))
    counterSamples = 0
    for sampleIndex in range(len(predProb)):
      correctLabel = yTrue[sampleIndex] # the semantic label
      if not correctLabel in classToIndex: # the label wasn't in the train data - ignore!
        continue
      counterSamples +=1 
      correctLabelIndex = classToIndex[correctLabel] # column index  of label in predProb matrix
      actualProb[sampleIndex][correctLabelIndex]=1.0
      logLossPerSample[sampleIndex] = metrics.log_loss(np.array([actualProb[sampleIndex]]), np.array([predProb[sampleIndex]])) * \
        sampleWeights[sampleIndex] 
    logLossAverage = np.sum(logLossPerSample) / float(counterSamples)

    return (logLossAverage, logLossPerSample)
  
  @staticmethod
  def unmapClasses(predLabels, classToIndex):
    '''
    predLabels: list of predicted class-indices
    classToIndex: dictionary from class label to its index
                  Also see :func:`model.Model.mapClasses`
    returns: list with the original class label or each class-index in predLabels
    '''
    classesUniqueSorted = sorted(list(classToIndex.keys()))
    return np.array(list(map(lambda x: classesUniqueSorted[x], predLabels)))
  
  @staticmethod
  def mapClasses(yData):
    '''
    yData: 1D array with true class label per sample
    returns: a dictionary mapping each true class to an integer in [0,number classes].
            this can be used to transform the input before giving it to a learning 
            algorithm in which the classes need to be defined as such a range.
            Also see :func:`model.Model.unmapClasses`
    '''
    classesUniqueSorted = sorted(np.unique(yData))
    classToIndex = {}
    for i in range(len(classesUniqueSorted)):
      classToIndex[classesUniqueSorted[i]]=i
    return classToIndex
  
  @staticmethod
  def getTrainTestSplits(y, n, fractionTest):
    try:
      return StratifiedShuffleSplit(y, n_iter=n, test_size=fractionTest)
    except ValueError:
      return ShuffleSplit(y.shape[0], n_iter=n, test_size=fractionTest)
   
class DNNModel(Model):
  
  def __init__(self, hiddenLayers = [5, 10, 5], iterations=100):
    super(DNNModel, self).__init__()
    self.hiddenLayers = hiddenLayers
    self.iterations = iterations
    
  def learn(self, x, y, weights=[], verbose=False, classToIndex=None):
    
    self.classToIndex = Model.mapClasses(y) if classToIndex == None else classToIndex

    yNewClass = np.array(list(map(lambda x: self.classToIndex[x], y)))
    nrClasses = len(self.classToIndex)
      
    feature_columns = [ tf.feature_column.numeric_column( 'x', shape=x.shape[1])]
    trainWeights = np.ones(y.shape).reshape(y.shape[0],1) if len(weights) == 0 else weights.reshape(y.shape[0],1)
    trainInputFn = tf.estimator.inputs.numpy_input_fn(x={'x': x, 'weights' : trainWeights}, y=yNewClass.reshape(y.shape[0],1), shuffle=True, num_epochs=self.iterations)
     
    # uses L2 loss
    nnClassifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, hidden_units=self.hiddenLayers, n_classes=nrClasses,
                                                  weight_column_name='weights')
    
    nnClassifier.fit(input_fn=trainInputFn)
    
    testInputFn = tf.estimator.inputs.numpy_input_fn(x={'x': x, 'weights' : trainWeights}, y=yNewClass.reshape(y.shape[0],1), shuffle=False, num_epochs=1)
    predLabelsNewClass = np.asarray(list( nnClassifier.predict_classes(input_fn=testInputFn)))
    predLabels = self.unmapClasses(predLabelsNewClass, self.classToIndex)
    predProb = np.asarray(list(nnClassifier.predict_proba(input_fn=testInputFn)))
    
    (acc,F1) = self.getCommonMetricScores(y, predLabels, sampleWeights = weights)
    (logLossAverage, logLossPerSample) = self.getLogLoss(y, predProb, self.classToIndex, sampleWeights = weights)
    if verbose:
      print('Accuracy: {0:2.4f}, F1: {1:2.4f}'.format(acc, F1))
      print('Log loss (using prob) = {0:2.4f}'.format(logLossAverage))
    
    self.model = nnClassifier
 
    return (predLabels, predProb, self.classToIndex) 
  
  def newSamples(self):
    return self.newX

  def predict(self, x):
    if self.model == None:
      raise RuntimeError('no model learnt yet')
    self.newX = x
    inputFn = tf.estimator.inputs.numpy_input_fn(x={'x': x}, shuffle=False)
    predLabelsNewClass = np.asarray(list( self.model.predict_classes(input_fn=inputFn)))
    predLabels = self.unmapClasses(predLabelsNewClass, self.classToIndex)
    predProb = np.asarray(list(self.model.predict_proba(input_fn=inputFn)))
    return (predLabels, predProb)
  
class XgboostModel(Model):
  
  def __init__(self, params=None, nrIterations=10):
    super(XgboostModel, self).__init__()
    
    defaultParam = {}
    defaultParam['objective'] = 'multi:softprob' 
    defaultParam['eta'] = 0.1   
    defaultParam['max_depth'] = 6
    defaultParam['silent'] = 1
    defaultParam['nthread'] = cpu_count() if cpu_count() != None else 4
    if params != None:
      defaultParam.update(params)
    self.param = defaultParam
    self.nrIterations = nrIterations
    
    
  def learn(self, x, y, weights=[], verbose=False, classToIndex=None):
    self.classToIndex = Model.mapClasses(y) if classToIndex == None else classToIndex
    
    yNewClass = np.array(list(map(lambda x: self.classToIndex[x], y)))
    
    self.param['num_class'] = len(self.classToIndex)
    
    trainWeights = np.ones(y.shape) if len(weights) == 0 else weights
    xgTrain = xgb.DMatrix(x, label=yNewClass, weight=trainWeights)
    xgTest = xgb.DMatrix(x, label=yNewClass, weight=trainWeights)
    evalList = [(xgTrain, 'train'), (xgTest, 'test')]

    bdtModel = xgb.train(self.param, xgTrain, self.nrIterations, evalList)

    predProb = bdtModel.predict(xgTest)
    predLabelsNewClass = np.argmax(predProb, axis=1)
    predLabels = self.unmapClasses(predLabelsNewClass, self.classToIndex)
    
    self.model = bdtModel
    
    (acc,F1) = self.getCommonMetricScores(y, predLabels, sampleWeights=trainWeights)
    (logLossAverage, logLossPerSample) = self.getLogLoss(y, predProb, self.classToIndex, sampleWeights=trainWeights)
    if verbose:
      print('Accuracy: {0:2.4f}, F1: {1:2.4f}'.format(acc, F1))
      print('Log loss (using prob) = {0:2.4f}'.format(logLossAverage))
    
    return (predLabels, predProb, self.classToIndex)
      
  def predict(self, x):
    if self.model == None:
      raise RuntimeError('no model learnt yet')
    testSet = xgb.DMatrix(x)
    predProb =  self.model.predict(testSet)
    predLabelsNewClass = np.argmax(predProb, axis=1)
    predLabels = self.unmapClasses(predLabelsNewClass, self.classToIndex)
    return (predLabels, predProb)