import numpy as np 
import sklearn.linear_model
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import pandas as pd 
from sklearn import svm
import math
import collections
from sklearn import preprocessing
from csv import DictReader, DictWriter
from sklearn.feature_selection import SelectFromModel
from sklearn import ensemble
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import mean_squared_error
import sklearn
import csv
from numpy import unravel_index
from sklearn.cross_validation import train_test_split

def main():
	print('The scikit-learn version is {}.'.format(sklearn.__version__))

	train  = pd.read_csv('LearningData.csv')

	test  = pd.read_csv('TestData.csv')

#	print train.columns

#	print len(train.columns)

#	print len(train)

#	print len(test)

#	print train.values[0]

	numerictraindf = train[train.columns[4:41]].apply(pd.to_numeric, errors='coerce')

#	print numerictraindf.dtypes

	firstcolumntrain = train[train.columns[0:1]].apply(pd.to_numeric, errors='coerce')

#	print firstcolumntrain.dtypes
#	print firstcolumntrain.columns

	stringtraindf = train[train.columns[1:4]]

#	print stringtraindf.dtypes
#	print stringtraindf.columns
#	print len(stringtraindf.columns)
#	print len(numerictraindf.columns)
#	print firstcolumntrain.values[0]

	count = 1

	EncodeStringData = []

	for each in stringtraindf.columns:
		if count == 1:
			EncodeStringData = stringtraindf[each]
			count = 2
		else:
			EncodeStringData = np.column_stack((EncodeStringData,stringtraindf[each]))
	
#	print EncodeStringData.shape

	stringdatareshape = np.reshape(EncodeStringData,(len(train)*len(stringtraindf.columns),1))

#	print stringdatareshape.shape
  
	learningData = sklearn.preprocessing.LabelEncoder()
   
	trainTranformed = learningData.fit_transform(stringdatareshape)

#	print max(trainTranformed)
#	print test.values[0]    

	numerictestdf = test[test.columns[4:41]].apply(pd.to_numeric, errors='coerce')    
	
#	print numerictestdf.dtypes    
	
	firstcolumntest = test[test.columns[0:1]].apply(pd.to_numeric, errors='coerce')    
	
#	print firstcolumntest.dtypes
#	print firstcolumntest.columns    
	
	stringtestdf = test[test.columns[1:4]]    
	
#	print stringtestdf.dtypes
#	print stringtestdf.columns
#	print len(stringtestdf.columns)    
#	print len(numerictestdf.columns)    
#	print firstcolumntest.values[0]
   
	count = 1    
	EncodeTestStringData = []    
	for each in stringtestdf.columns:
	   if count == 1:
		   EncodeTestStringData = stringtestdf[each]
		   count = 2
	   else:
		   EncodeTestStringData = np.column_stack((EncodeTestStringData,stringtestdf[each]))    

#	print EncodeTestStringData.shape    #Changes the csv data into a one dimensional array
	
	stringTestDataReshape = np.reshape(EncodeTestStringData,(len(test)*len(stringtestdf.columns),1))
   
#	print stringtestdf.shape
#	print stringTestDataReshape.shape
   
	TransformedStringTestData = learningData.transform(stringTestDataReshape)

	TransformedStringData = np.reshape(trainTranformed,(len(train),3))

#	print TransformedStringData.shape
   
	TransformStringTestData = np.reshape(TransformedStringTestData,(len(test),3))

#	print TransformStringTestData.shape

	newtraindf = pd.concat([firstcolumntrain, numerictraindf], axis=1)

	newTestdf = pd.concat([firstcolumntest, numerictestdf], axis=1)

#	print newtraindf.shape
#	print newTestdf.shape

	#Reassembling numeric data - training data set

	count = 1    
	FinalNumerictrain = []    
	for each in newtraindf.columns:
	   if count == 1:
		   FinalNumerictrain = newtraindf[each]
		   count = 2
	   else:
		   FinalNumerictrain = np.column_stack((FinalNumerictrain,newtraindf[each]))

	#Reassembling numeric data - training data set

	FinalNumerictest = []    
	count = 1
	for each in newTestdf.columns:
	   if count == 1:
		   FinalNumerictest = newTestdf[each]
		   count = 2
	   else:
		   FinalNumerictest = np.column_stack((FinalNumerictest,newTestdf[each]))

	#Reassembling complete data set with string data

	FinalTrainX = np.column_stack((FinalNumerictrain, TransformedStringData)) 
	FinalTestX = np.column_stack((FinalNumerictest, TransformStringTestData))

	print FinalTrainX.shape
	print FinalTestX.shape

	# Generating labels for training data attacks

	labeldata = []

	for each in train['Attack Category']:
		if not each in labeldata:
			labeldata.append(each)

	print labeldata
	print len(labeldata)

	#Generating list of y values for training data set

	y_train = []
	for each in train['Attack Category']:
		y_train.append(int(labeldata.index(each) + 1))

	# print len(y_train)
	# print labeldata[0]
	# print labeldata[1]
	# print labeldata[2]
	# print labeldata[3]
	# print labeldata[4]
	# print collections.Counter(y_train)

	#Splitting the datat training set into a training set and a dummy test data set
	#Train_test_split(training data set, output values, % of data set to use as 'test', random seed)

	X_dummytrain, X_dummytest, y_dummytrain, y_dummytest = train_test_split(FinalTrainX, y_train, test_size=0.0, random_state=42)

	# print X_dummytrain.shape
	# print X_dummytest.shape
	# print len(y_dummytrain)
	# print collections.Counter(y_dummytrain)
	# print len(y_dummytest)
	# print collections.Counter(y_dummytest)
	
	#Machine learning algorithm

	classifiers = [
	ensemble.GradientBoostingClassifier(n_estimators = 500, learning_rate = 0.1, max_depth = 10 , random_state = 0)]
	
	#Code for testing the algorithm against the training data set

	"""
	for clf in classifiers:
		estimate = clf.fit(X_dummytrain,y_dummytrain)
		predictions = estimate.predict(X_dummytest)

		print accuracy_score(y_dummytest, predictions)
	"""
	#Generating y output list for true test data set
	
	y_test = []
	for each in test['Attack Category']:
		y_test.append(int(labeldata.index(each) + 1))
	
	#Using the complete training data set to to predict attacks on the test data set

	for clf in classifiers:
		estimate = clf.fit(X_dummytrain, y_dummytrain)
		predictions = estimate.predict(FinalTestX)

		print accuracy_score(y_test, predictions)

if __name__ == "__main__":
	main()