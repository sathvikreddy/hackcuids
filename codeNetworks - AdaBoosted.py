#import argparse
import numpy as np 
import sklearn.linear_model
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import pandas as pd 
from sklearn import svm
#import re
import math
import collections
from sklearn import preprocessing
from csv import DictReader, DictWriter
#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
#from sklearn.linear_model import SGDClassifier
from sklearn import ensemble
from sklearn.tree import DecisionTreeClassifier
#from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import mean_squared_error
import sklearn
import csv
#from sklearn.preprocessing import Imputer
from numpy import unravel_index
#from sklearn.base import TransformerMixin
from sklearn.cross_validation import train_test_split




def End ():
	print "Question Finish"



if __name__ == "__main__":
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
#

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


	#trainTranformed = learningData.fit_transform(stringdatareshape)
	#Outputs the classes from the learningData list
	
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

#	print "HI"

	
   
	TransformStringTestData = np.reshape(TransformedStringTestData,(len(test),3))    #Outputs the classes from the learningData list
   
	# print TransformStringTestData.shape



	newtraindf = pd.concat([firstcolumntrain, numerictraindf], axis=1)

	newTestdf = pd.concat([firstcolumntest, numerictestdf], axis=1)

	# print newtraindf.shape
	# print newTestdf.shape

	count = 1    
	FinalNumerictrain = []    
	for each in newtraindf.columns:
	   if count == 1:
		   FinalNumerictrain = newtraindf[each]
		   count = 2
	   else:
		   FinalNumerictrain = np.column_stack((FinalNumerictrain,newtraindf[each]))


	FinalNumerictest = []    
	count = 1
	for each in newTestdf.columns:
	   if count == 1:
		   FinalNumerictest = newTestdf[each]
		   count = 2
	   else:
		   FinalNumerictest = np.column_stack((FinalNumerictest,newTestdf[each]))


	FinalTrainX = np.column_stack((FinalNumerictrain, TransformedStringData)) 
	FinalTestX = np.column_stack((FinalNumerictest, TransformStringTestData))

	print FinalTrainX.shape
	print FinalTestX.shape

	labeldata = []

	for each in train['Attack Category']:
		if not each in labeldata:
			labeldata.append(each)

	print labeldata
	print len(labeldata)

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

	X_dummytrain, X_dummytest, y_dummytrain, y_dummytest = train_test_split(FinalTrainX, y_train, test_size=0.0, random_state=42)


	# print X_dummytrain.shape
	# print X_dummytest.shape
	# print len(y_dummytrain)
	# print collections.Counter(y_dummytrain)
	# print len(y_dummytest)
	# print collections.Counter(y_dummytest)

	classifiers = [
	AdaBoostClassifier(DecisionTreeClassifier(max_depth=4), n_estimators=600)]

	#,AdaBoostClassifier(DecisionTreeClassifier(max_depth=4), n_estimators=600)
	"""
	for clf in classifiers:
		estimate = clf.fit(X_dummytrain,y_dummytrain)
		predictions = estimate.predict(X_dummytest)

		print accuracy_score(y_dummytest, predictions)
	"""
	#labeldata = [] ####adding what I *think* is the code we need to test our results

	# for each in train['Attack Category']:
	# 	if not each in labeldata:
	# 		labeldata.append(each)

	# print labeldata
	# print len(labeldata)

	y_test = []
	for each in test['Attack Category']:
		y_test.append(int(labeldata.index(each) + 1))
	
	for clf in classifiers:
		estimate = clf.fit(X_dummytrain, y_dummytrain)
		predictions = estimate.predict(FinalTestX)

		print accuracy_score(y_test, predictions)

	print A.shape

"""


	#SVM
	#ExtraTrees
	#Adaboost
	#Random
	#DecisionTreeradientBoosting

	rng = np.random.RandomState(1)


	#precision = precision_score(y_test, predictions, [-1, 1])
	#recall = recall_score(y_test, predictions, [-1, 1])
	#auc_score = roc_auc_score(y_test, predictions, None)

	classifiers = [
	DecisionTreeClassifier(max_depth=10),
	LinearSVC(C=0.01, penalty="l1", dual=False),
	ensemble.ExtraTreesClassifier(n_estimators = 100),
	ensemble.RandomForestClassifier(max_depth=10, n_estimators=100)]

	computeclassifiers = [
	ensemble.GradientBoostingClassifier(n_estimators = 500, learning_rate = 0.1, max_depth = 10 , random_state = 0)]

	scores = np.zeros((4, 4))

	precmatrix = np.zeros((4,4))
	recmatrix = np.zeros((4,4))

	#LinearSVC(C=0.01, penalty="l1", dual=False),
	#ensemble.ExtraTreesClassifier(n_estimators = 100),
	#ensemble.RandomForestClassifier(max_depth=10, n_estimators=100)

	#LinearSVC(C=0.01, penalty="l1", dual=False),
	#ensemble.ExtraTreesClassifier(n_estimators = 100, random_state = 0),
	#DecisionTreeClassifier(max_depth=10),
	#ensemble.AdaBoostClassifier(DecisionTreeClassifier(max_depth=10), n_estimators=300),

	# lsvc = ensemble.ExtraTreesClassifier(n_estimators = 100, random_state = 0).fit(X_dummytrain,y_dummytrain)
	# model = SelectFromModel(lsvc, prefit = True)

	# Train_new = model.transform(X_dummytrain)
	# print Train_new.shape
	# newindices = model.get_support(True)

	# FinalTrainLessFeature = X_dummytrain[np.ix_(np.arange(40000), newindices)]
	# FinalTestLessFeature = X_dummytest[np.ix_(np.arange(10000), newindices)]

	# print FinalTrainLessFeature.shape
	# print FinalTestLessFeature.shape


	# estimate = ensemble.GradientBoostingClassifier(n_estimators = 50, learning_rate = 0.1, max_depth = 10 , random_state = 0).fit(FinalTrainLessFeature,y_dummytrain)

	# predictions = estimate.predict(FinalTestLessFeature)

	# print "ExtraTrees"
	# print predictions.shape
	# print y_dummytest.shape
	# precision = precision_score(y_dummytest, predictions)
	# print precision
	# recall = recall_score(y_dummytest, predictions, [-1, 1])
	# print recall
	# auc_score = roc_auc_score(y_dummytest, predictions)
	# print auc_score

	


	
	# lsvc = ensemble.RandomForestClassifier(max_depth=10, n_estimators=100)
	# model = SelectFromModel(lsvc, prefit = True)

	# Train_new = model.transform(X_dummytrain)
	# print Train_new.shape
	# newindices = model.get_support(True)

	# FinalTrainLessFeature = X_dummytrain[np.ix_(np.arange(40000), newindices)]
	# FinalTestLessFeature = X_dummytest[np.ix_(np.arange(10000), newindices)]

	# print FinalTrainLessFeature.shape
	# print FinalTestLessFeature.shape


	# estimate = ensemble.GradientBoostingClassifier(n_estimators = 500, learning_rate = 0.1, max_depth = 10 , random_state = 0).fit(FinalTrainLessFeature,y_dummytrain)

	# predictions = estimate.predict(FinalTestLessFeature)

	# precision = precision_score(y_dummytest, predictions, [-1, 1])
	# recall = recall_score(y_dummytest, predictions, [-1, 1])
	# auc_score = roc_auc_score(y_dummytest, predictions, None)

	# print "RandomTrees"
	# print precision
	# print recall_score
	# print auc_score

	# print A.shape	


	i = 0 

	for clf in classifiers:
		lsvc = clf.fit(X_dummytrain,y_dummytrain)
		model = SelectFromModel(lsvc, prefit = True)

		Train_new = model.transform(X_dummytrain)
		print Train_new.shape
		newindices = model.get_support(True)

		FinalTrainLessFeature = X_dummytrain[np.ix_(np.arange(40000), newindices)]
		FinalTestLessFeature = X_dummytest[np.ix_(np.arange(10000), newindices)]

		print FinalTrainLessFeature.shape
		print FinalTestLessFeature.shape
	

		j =0
		print collections.Counter(y_dummytest)
		print clf
		print newindices

		for cllf in computeclassifiers:
			rng = np.random.RandomState(1)

			estimate = cllf.fit(FinalTrainLessFeature,y_dummytrain)

			predictions = estimate.predict(FinalTestLessFeature)

			scores[i][j] = accuracy_score(y_dummytest,predictions) 
			precision = precision_score(y_dummytest, predictions)
			print precision
			precmatrix[i][j] = precision 
			recall = recall_score(y_dummytest, predictions, [-1, 1])
			print recall
			recmatrix[i][j] = recall
			print roc_auc_score(y_dummytest, predictions, None)

			j = j + 1

		FinalTestLessFeature = []
		FinalTrainLessFeature = []
		i = i + 1

	
	



	print scores
	print precmatrix
	print recmatrix

	print A.shape

	i , j = unravel_index(scores.argmax(), scores.shape)


	lsvc = classifiers[i].fit(Full_X_TrainData,labels)
	
	model = SelectFromModel(lsvc, prefit = True)

	Train_new = model.transform(Full_X_TrainData)
	print Train_new.shape
	newindices = model.get_support(True)

	FinalTrainLessFeature = Full_X_TrainData[np.ix_(np.arange(lengthTrain), newindices)]
	FinalTestLessFeature = Full_X_TestData[np.ix_(np.arange(lengthTest), newindices)]

	print FinalTrainLessFeature.shape
	print FinalTestLessFeature.shape

	rng = np.random.RandomState(1)

	finalestimate = classifiers[j].fit(FinalTrainLessFeature,labels)

	predictions = finalestimate.predict(FinalTestLessFeature)
	

	print "In writePredictions"
	o = DictWriter(open("predictions1tocheck.csv", 'w'),["target"])
	for y_val in predictions:
		o.writerow({'target': y_val})

	End()
"""	



