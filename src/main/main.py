import os, sys
from collections import Counter
from ggplot import *
from preprocess import TripTypeDataSet, DataSet
from preprocess import mergeDescriptions
from copy import deepcopy
import numpy as np
from model import XgboostModel, DNNModel
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
from adaboost import AdaBoost
  

#random.seed(13132)
#np.random.seed(13132)

def getGeneralizedDescriptions(df, verbose=False):
  descriptions = list(filter(lambda x: str(x) !='nan', list(set(df.ix[:,'DepartmentDescription']))))
  newDescriptionToDescriptions = mergeDescriptions(descriptions)
  if verbose:
    print('new description to old description map:')
    for k,v in newDescriptionToDescriptions.items():
      print (k,'->',v)
  return newDescriptionToDescriptions

def getCommonManufacturer(df, verbose=False):
  tempUpc = DataSet.removeRowsWithNulls(df)['Upc'].apply(lambda x: '{0:.0f}'.format(x))
  mapStrLen = list(map(lambda x: (len(str(x)), x,), tempUpc))
  if verbose:
    maxUpc = max(mapStrLen) # 12
    minUpc = min(mapStrLen) # 4
    print ('max upc length: {0:d}, min upc length: {1:d}'.format(maxUpc, minUpc))
  tempUpc = df['Upc'].apply(lambda x: '{0:012.0f}'.format(x))
  c = Counter(tempUpc.apply(lambda x: x[1:6]))
  mostCommonManufacturer = c.most_common(20)
  if verbose:
    print('common manufacturer codes, their percentage:')
    for k,v in mostCommonManufacturer:
        print (k, '{:2.2f}'.format(v/tempUpc.shape[0]*100))
    c = Counter(tempUpc.apply(lambda x: x[6:11]))
    # as you can see most common products are low occurrence
    mostCommonProduct = c.most_common(20)
    print('common product codes, their percentage:')
    for k,v in mostCommonProduct:
        print (k, '{:2.2f}'.format(v/tempUpc.shape[0]*100))
  mostCommonManufacturer = [x[0]  for x in mostCommonManufacturer]
  return mostCommonManufacturer

def getMostCommonFLN(df, verbose=False):
  tempFLN= DataSet.removeRowsWithNulls(df)['FinelineNumber'].apply(lambda x: '{0:04.0f}'.format(x))
  c = Counter(tempFLN)
  mostCommonFLN = c.most_common(20)
  if verbose:
    print('common FLNs, their percentage:')
    for k,v in mostCommonFLN:
        print (k, '{:2.2f}'.format(v/tempFLN.shape[0]*100))
  mostCommonFLN =  [x[0] for x in mostCommonFLN]
  return mostCommonFLN


def preprocessData(trainFile, testFile, trainOuputFile, testOutputFile, maxTripTypesToSampleTrain=None, maxTripTypesToSampleTest=None, prune=True):
  trainSet = TripTypeDataSet(trainFile, prune=prune)
  
  translateColumnMap = {'DepartmentDescription':'DDS', 'Weekday':'WD'}
  oneHotColumns = ['DDS', 'WD']
  newDescriptionToDescriptions = getGeneralizedDescriptions(trainSet.df)
  mostCommonManufacturer = getCommonManufacturer(trainSet.df)
  mostCommonFLN = getMostCommonFLN(trainSet.df)
  
  trainSet.restructure(translateColumnMap, oneHotColumns, newDescriptionToDescriptions, mostCommonManufacturer, mostCommonFLN, maxTripTypesToSample=maxTripTypesToSampleTrain)
  trainSet.toCsv(trainOuputFile)
   
  colNamesTrain = deepcopy(trainSet.colNames)
  colNamesTrain.remove('TripType')
  
  testSet = TripTypeDataSet(testFile, prune=prune)
  testSet.restructure(translateColumnMap, oneHotColumns, newDescriptionToDescriptions, mostCommonManufacturer, mostCommonFLN, maxTripTypesToSample=maxTripTypesToSampleTest,
                      dummyColumns=colNamesTrain)
  testSet.toCsv(testOutputFile)
  
  print (trainSet.colNames)
  print (testSet.colNames)

def plotTripTypeCount(trainSet):
  p = ggplot( aes(x='TripType'), data=trainSet.df) 
  p + geom_bar(stat="bin")
  print(p)
  
  
def plotFeatureCorrs(trainSet):
  colormap = plt.cm.viridis
  seaborn.set(font_scale=0.5)
  plt.figure(figsize=(25,25))
  plt.title('Pearson Correlation of Features', y=1.05, size=15)
  seaborn.heatmap(trainSet.df.astype(float).corr(),
                  linewidths=0.1, 
                  square=False, cmap=colormap, 
                  linecolor='white', annot=False)
  plt.show()
  
def writeTestSet(visitNumbers, classToIndex, predProbPerSample, outfile):
  indexToClass = {v:k for k,v in classToIndex.items()}
  
  if len(indexToClass) < predProbPerSample.shape[1]:
    print('not all classes were predicted')
  cols = ['VisitNumber'] + ['TripType_'+str(indexToClass[x]) for x in range(len(indexToClass))] + ['TripType']
  df = pd.DataFrame(columns=cols)
  i = 0
  for visitNumber in visitNumbers:
    l = ['{:.4f}'.format(x) for x in  list(predProbPerSample[i,]) ]
    classL = [indexToClass[np.argmax(l)]]
    il = [visitNumber] + l + classL
    df.loc[i] = il
    i += 1
    if ((i % 500) == 0):
      print('writing row {:d} of {:d}'.format(i, len(visitNumbers)))
  df[['VisitNumber']] =  df[['VisitNumber']].astype(int)
  df.sort_values('VisitNumber', inplace=True)
  df.to_csv(outfile, index=False)
  print ('output saved to {}'.format(os.path.realpath(outfile)))

if __name__=="__main__":
  print("hi")

  datafolder = "../../data"
  
  trainFile = os.path.join(datafolder, "train.csv")
  testFile= os.path.join(datafolder, "test.csv")
  
  trainFilePP = os.path.join(datafolder, "trainsetpp2.csv")
  testFilePP = os.path.join(datafolder, "testsetpp2.csv")
  
  # uncomment this if you don't have trainFilePP and testFilePP yet.
  # WARNING: can take a long time
  #preprocessData(trainFile, testFile, trainFilePP, testFilePP, prune=False)#, maxTripTypesToSampleTrain=1000, maxTripTypesToSampleTest=500)
  #preprocessData(trainFile, testFile, trainFilePP, testFilePP, prune=False)
  
  trainSet = TripTypeDataSet(trainFilePP, prune=False)
  testSet = TripTypeDataSet(testFilePP, prune=False)
  
  print(trainSet.nrows)
  # plots how often each triptype occurs in the data 
  #plotTripTypeCount(trainSet)
  
  # plots heatmap of feature correlations
  #plotFeatureCorrs(trainSet)
  
  # get raw train and test matrices
  yTrain = np.asarray(trainSet.df['TripType'])
  xTrain = np.asanyarray(trainSet.df.drop(['TripType', 'VisitNumber'], axis=1))
  xTest = np.asanyarray(testSet.df.drop(['VisitNumber'], axis=1))
  # sample for now to speed up training. NOTE: should ideally happen inside adaboost class, using bagging on top
  sampledIndices = np.random.choice(trainSet.nrows-1, min(10000, trainSet.nrows-1), replace=False)
  print(xTrain.shape)
  xTrain = xTrain[sampledIndices,]
  yTrain = yTrain[sampledIndices]

  
  # uncomment this for testing the models separately
#   m1 = XgboostModel(params={'max_depth': 3})
#   m1.learn(xTrain, yTrain)
#   predLabels = m1.predict(xTest)
  
#   m2 = DNNModel(iterations=500)
#   m2.learn(xTrain, yTrain)
#   predLabels = m2.predict(xTest)
#   
#   print('predLabels:',predLabels)b
#   print(list(testSet.df['VisitNumber']))
  
  
  # set up modellist for adaboost
  modelList = [
    XgboostModel(params={'max_depth': 3,  'eval_metric': 'mlogloss'}, nrIterations=3)]
   # DNNModel(iterations=400)]
  nrCopies=40
  origLen = len(modelList)
  for i in range(nrCopies):
    for reference in modelList[:origLen]:
      modelList.extend([deepcopy(reference)])
  print(modelList)
               
               
  #train adaboost
  adaBoost = AdaBoost(modelList)
  trainTestSplits = 4
  (accuracy, F1, logLoss) = adaBoost.crossValidate(xTrain, yTrain, trainTestSplits, 0.2, verbose=True)
  print('avg acc: {:2.4f}, avg F1: {:2.4f}, avg. log loss: {:2.4f} based on {:d} tt splits'.format(accuracy, F1, logLoss, trainTestSplits))
  #(predLabels, predProb, classToIndex) = adaBoost.learn(xTrain, yTrain, verbose=True)
  #predLabelsTest, predProbTest = adaBoost.predict(xTest) 
  
  # uncomment to write the predictions for the test set to a file 
  #writeTestSet(list(testSet.df['VisitNumber']), classToIndex, predProbTest, os.path.join(datafolder, 'testpred.csv'))
  
  
  
  