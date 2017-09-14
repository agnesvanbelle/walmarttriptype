import pandas as pd
import os 
from collections import defaultdict as dd
from collections import Counter
from itertools import combinations
import re
from nltk.corpus import stopwords
from multiprocessing import Pool, TimeoutError, cpu_count

stopwordsEnglish = set(stopwords.words('english'))


class DataSet(object):
  
  def __init__(self, filename):
    self.filename = os.path.realpath(filename)
    self.df = pd.read_csv(filename, index_col=False)
    self.nrows = self.df.shape[0]
    print('read {:d} rows'.format(self.nrows))
    self.__updateColumnStats__()
  
  def __updateColumnStats__(self):
    self.colNames = list(self.df)
    self.ncols = len(self.colNames)
  
  def toCsv(self, filename):
    self.df.to_csv(filename, index=False)
    print('wrote dataset to', os.path.realpath(filename))
    
  @staticmethod
  def removeRowsWithNulls(df, verbose=False):
    nrows = df.shape[0]
    nulls = {x:sum(pd.isnull(df[x])) for x in list(df)}
    nrDropped = 0
    for colName, _nrNull in nulls.items():
      indicesNull =  df[pd.isnull(df[colName])].index
      df.drop(indicesNull, inplace=True)
      nrDropped += len(indicesNull)
    if verbose:
      print('null values per column:',nulls)
      print('dropped {0:d} rows - {1:2.4f}%'.format(nrDropped, nrDropped / (nrows / 100.0) ))
    return df
  
  
class TripTypeDataSet(DataSet):
  
  def __init__(self, filename, prune=True):
    super(TripTypeDataSet, self).__init__(filename)
    
    if prune:
      self.df.drop(self.df.index[10000:], inplace=True)    # prune
      self.__updateColumnStats__()
    self.nrVisits = len(set(self.df['VisitNumber'])) 
    
    print('read data in',os.path.realpath(filename))
  
                     
  def restructure(self, translateColumnMap, oneHotColumns, generalizedDescriptionToDescriptions, mostCommonManufacturers, mostCommonFLN, dummyColumns=[],
                  maxTripTypesToSample=None):
    if self.nrows == self.nrVisits:
      print ('dataset already restructured.')
      return
    print('restructuring {0:s}...'.format(self.filename))
    # remove rows with na, nan, null
    self.df = DataSet.removeRowsWithNulls(self.df)
    self.nrows = self.df.shape[0]
    
    # rename columns
    self.df.rename(columns=translateColumnMap, inplace=True)
    self.__updateColumnStats__()
    
    # add one-hot encoding for specified columns
    dfOneHotEncoded = pd.get_dummies(self.df[oneHotColumns])
    for c in list(dfOneHotEncoded):
      self.df[c] = dfOneHotEncoded[c]
    self.__updateColumnStats__()
    
    # add generalized department descriptions
    self.__addGeneralizedDepartmentDescriptions__('DDS_', generalizedDescriptionToDescriptions)
    
    # encode Upc and FLN
    self.__encodeUpc__(mostCommonManufacturers)
    self.__encodeFLN__(mostCommonFLN)
    
    # now we can drop the original columns that should be one-hot encoded
    self.df.drop(oneHotColumns, axis=1, inplace=True)
    # now we can drop Upc and FLN columns
    self.df.drop(['FinelineNumber', 'Upc'], axis=1, inplace=True)
    self.__updateColumnStats__()
    
    # transform VisitNumber to string
    if 'VisitNumber' in self.colNames:
      self.df['VisitNumber'] = self.df['VisitNumber'].apply(lambda x: '{0:d}'.format(x))
    
    # reshape so that each trip is one row
    self.__reshapePerTrip(maxTripTypesToSample)
    
    # make sure train/test set have same column names in same order
    for c in dummyColumns:
      if not c in self.df.columns:
        self.df[c] = 0    
    if dummyColumns:
      for c in self.colNames:
        if not c in dummyColumns:
          self.df.drop(c, axis=1, inplace=True)
    self.df = self.df.reindex_axis(sorted(self.df.columns), axis=1)
    self.__updateColumnStats__()
    
    
  '''
  From wikipedia:
  12-digit UPC-A numbering schema lLLLLLRRRRRr, where l denotes number system digit and r check digit.
  (...)
  The LLLLL digits are the manufacturer code (assigned by local GS1 organization), and the RRRRR digits are the product code.
  '''
  def __encodeUpc__(self, mostCommonManufacturer):
    self.df['Upc'] =  self.df['Upc'].apply(lambda x: '{0:012.0f}'.format(x))
    self.df['manucode_other'] =1
    for code in mostCommonManufacturer:
      newDescriptionName = 'manucode_'+code
      self.df[newDescriptionName] = 0   
      rowsManuCode = self.df[self.df['Upc'].apply(lambda x:x[1:6] == code)].index
      self.df.loc[rowsManuCode, 'manucode_other'] = 0
      self.df.loc[rowsManuCode,newDescriptionName] = 1
    self.__updateColumnStats__()
    
  def __encodeFLN__(self, mostCommonFLN):
    self.df['FinelineNumber'] = self.df['FinelineNumber'].apply(lambda x: '{0:04.0f}'.format(x))
    self.df['FLN_other'] =1
    for code in mostCommonFLN:
      newDescriptionName = 'FLN_'+code
      self.df[newDescriptionName] = 0   
      rowsCode = self.df[self.df['FinelineNumber'] == code].index
      self.df.loc[rowsCode, 'FLN_other'] = 0
      self.df.loc[rowsCode,newDescriptionName] = 1
    self.__updateColumnStats__()
    
  def __addGeneralizedDepartmentDescriptions__(self, DDSoneHotCodePrefix, newDescriptionToDescriptions):
    for newDescription, oldDescriptions in newDescriptionToDescriptions.items():
      newDescriptionName = 'DDSG_'+newDescription
      self.df[newDescriptionName] = 0
      for oldDescription in oldDescriptions:        
        colName = DDSoneHotCodePrefix+oldDescription
        #print('oldDescription:{}, col name:{}, self.colNames:{}'.format(oldDescription, colName, self.colNames))
        if colName in self.colNames:
          self.df.loc[self.df[colName]==1,newDescriptionName] = 1
    self.__updateColumnStats__()

  @staticmethod
  def reshapeTrips(threadNumber, gb, retRows, colNames, start, end, detailedInfo=False, maxIter=None):
    
    dfNew = pd.DataFrame()
    gkeys = list(gb.groups.keys())
    i = 0
    for k in gkeys[start:min(end, len(gkeys))]:
      group = gb.get_group(k)

      if detailedInfo:
        relDDS =  Counter(dict( (group[ [col for col in list(group) if col.startswith('DDS')]]!=0).sum())) + Counter()
        relFLN =  Counter(dict( (group[ [col for col in list(group) if col.startswith('FLN')]]!=0).sum())) + Counter()
        relManucode = Counter(dict( (group[ [col for col in list(group) if col.startswith('manucode')]]!=0).sum())) + Counter()
        
        boughtAndReturned = {}
        allKeys = set(list(relDDS.keys())+list(relFLN.keys()) + list(relManucode.keys()))
        for k in allKeys:
            boughtAndReturned[k] = sum(group.loc[(group[k]>0) & (group['ScanCount']>0), 'ScanCount'])
            boughtAndReturned[k+'_r']  = abs(sum(group.loc[(group[k]>0) & (group['ScanCount']<0), 'ScanCount']))
      s = pd.DataFrame(group.sum(axis=0, numeric_only=True)).transpose()    
      ssoIndex = group[group['ScanCount']<0].index
      if 'TripType' in colNames:
        s['TripType'] = list(group[['TripType']]['TripType'])[0]
      s['VisitNumber'] =  k
      s['numberReturns'] = sum(group['ScanCount']<0)
      s['amountReturns'] = sum(group.loc[ssoIndex,'ScanCount'])
      if detailedInfo:
        for x in retRows:
            s[x] = 0
        for k,v in boughtAndReturned.items():
            s[k] = v
      dfNew = dfNew.append(s, ignore_index=True)
      i +=1
      if (i % 500) == 0:
          print('reshaping per trip: trip {:d} of {:d}, thread {}'.format(i, (end-start), threadNumber))
      if maxIter != None and i==maxIter:
          print('breaking because a maximum to sample was set: {}'.format(maxIter))
          break
    return dfNew
  
  # if you set detailedInfo=True, more information will be included per row. 
  def __reshapePerTrip(self, maxIter, detailedInfo=False):
    retRows =  [x+'_r' for x in self.colNames if x.startswith('DDS') or x.startswith('FLN') or x.startswith('manucode')]

    gb = self.df.groupby('VisitNumber')
    nrGroups = len(gb)
    if maxIter != None:
      nrGroups = maxIter
    nrCpus = cpu_count() if cpu_count() != None else 4
    interval = int(nrGroups / nrCpus)
    slices = []
    for i in range(nrCpus):
      start = i*interval
      slices.append((start, (start+interval) if i < (nrCpus-1) else nrGroups,))
    
    pool = Pool(nrCpus)
    results=[-1]*nrCpus
    for i in range(nrCpus):
      results[i] = pool.apply_async(reshapeTripsHelper, (i, gb, retRows, self.colNames, slices[i][0], slices[i][1], detailedInfo, maxIter))
    pool.close()
    pool.join()
    
    dfs = [-1]*nrCpus
    for i in range(nrCpus):
      dfs[i] = results[i].get()
    dfNew = pd.concat(dfs, ignore_index=True)
    
    print ('dfNew.shape:{}'.format(dfNew.shape))
    self.df = dfNew
    self.__updateColumnStats__()

# helper because apply_async doesn't work with (static) methods as functions
def reshapeTripsHelper(threadNumber, gb, retRows, colNames, start, end, detailedInfo=False, maxIter=None):
  return TripTypeDataSet.reshapeTrips(threadNumber, gb, retRows, colNames, start, end, detailedInfo, maxIter)

# given a list of input descriptions, output a dictionary mapping a new more general description 
# to a list of the original descriptions
# it belongs to. Generalizability is based on word overlap between descriptions. 
def mergeDescriptions(descriptions, minWordOverlap=1):
    newDescriptionToDescriptions = dd(set)
    combis = combinations(range(len(descriptions)), 2)
    descriptionsNormalizedSplitted = list(map(lambda x : set(normalizeDescription(x).split()), descriptions))
    for pair in combis:
        (descr1, descr2) = (descriptions[pair[0]], descriptions[pair[1]])
        (descr1Set, descr2Set) = (descriptionsNormalizedSplitted[pair[0]], descriptionsNormalizedSplitted[pair[1]])
        overlap = descr1Set.intersection(descr2Set)
        if len(overlap) >= minWordOverlap:
            newDescr = "_".join(overlap)
            newDescriptionToDescriptions[newDescr].update([descr1, descr2])          
    newDescriptionToDescriptions = {k:list(v) for k,v in newDescriptionToDescriptions.items()}
    return newDescriptionToDescriptions

def normalizeDescription(description):
    description = re.sub(r'[^a-zA-Z]+', ' ', description).lower()
    description = " ".join([x for x in description.split() if (x not in stopwordsEnglish) and len(x) > 1])
    return description.lower().strip()