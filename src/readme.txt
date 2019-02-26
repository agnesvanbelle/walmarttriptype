Dependencies:
Python 3.5 (included in Anaconda 4.2.0)
Python packages:
  tensorflow 1.2.1
  numpy 1.13.1
  xgboost 0.6
  scikit-learn 0.17.1
  pandas 0.18.1

Rationale:
Code starts at main.py.

## First I transformed the input data for training (see preprocess.py):
- remove rows with na/null (was a very small percentage of the train set)
- transform WeekDay and DepartmentDescription into a one-hot encoding
- add more generalized DepartmentDescriptions to the dataset, based on the overlap between existing DepartmentDescriptions. There where many department descriptions, and I saw that many were sort of the same (e.g. about clothing).
- encode UPC and FinelineNumber (FLN) in a one-hot encoding using their top-20 most frequent manufacturer-code (for UPC) and existing value (FLN). There is also a product code in the UPC but I saw that none of these occurred in more than a few percent of the samples.
- reshape everything so that each single row represents a single visit (VisitNumber). The idea behind this, is that each visit is a single instance for training/testing that should contain all information currently distributed over multiple rows. I initially did this, by calculating for each FLN/UPC/(generalized )DepartmentDescription the number of product purchases and returns, per visit. Because this has to be done with a loop, this takes a while to run. Currently, I do it a bit more simplied: just sum all rows per visit with each other (plus add separate columns for the number and amount of returns). It is done multithreadingly. 
After that the transformed train and test set are written to output csv files.

### Then making models (see models.py).
One based on a neural network with 3 hidden layers and L2 loss for classification and another one based on gradient boosted regression trees (BRTs) using the multiclass classfication error rate as cost function (but this can be changed by changing a parameter in the constructor).
These are supposed to be simple but powerful.
The generalizability of the NN model can be controlled by passing the hidden layer architecture as well as the number of epochs to its constructor.
The generalizability of the BRT model can be controlled by passing the number of iterations to its constructor as well as a parameter dictionary with e.g. the maximum tree depth in it.

### Then I made a custom ensemble meta-model using boosting. It implements real-valued multiclass AdaBoost, a.k.a. SAMME.R - that takes a list of the abovementioned models as input (see adaboost.py). The AdaBoost implementation in scikit is limited in the types of base models it can handle, so this implementation serves as an easier wrapper.
The idea is to boost the samples on which the previous model performed poorly when training the next model. The reason I chose for that is because there are many classes in the data set and adaboost is supposed to work well with classification problems; but I also thought that, because there are so many classes, that some model types would be better than others on predicting certain classes.





