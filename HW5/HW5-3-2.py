
from numpy import *
from sklearn import linear_model
import random 


def ShuffleDataSet(DataSet):
	random.shuffle(DataSet)
	return DataSet

def GetFeature(DataSet, num):
	Featurelist = []
	for vector in DataSet:
		Featurelist.append(vector[num])
	return Featurelist

def GetData(DataSet):
	Featurelist = []
	for vector in DataSet:
		Featurelist.append(vector[:-1])
	return Featurelist

def loadData(filename):
	vector = []
	with open(filename) as file:
		for line in file:
			vector.append(map(float,line.split()))
	return vector

TrainData = loadData('/Users/skylatitude/downloads/spam_polluted/train_feature.txt')
TrainLabel = loadData('/Users/skylatitude/downloads/spam_polluted/train_label.txt')
TestData = loadData('/Users/skylatitude/downloads/spam_polluted/test_feature.txt')
TestLabel = loadData('/Users/skylatitude/downloads/spam_polluted/test_label.txt')
for i in range(len(TrainData)):
	TrainData[i].append(TrainLabel[i])
TrainData = ShuffleDataSet(TrainData)
TrainLabel = GetFeature(TrainData,-1)
traindata = GetData(TrainData)
m = len(TrainData)
model = linear_model.LogisticRegression(penalty="l1")
model.fit(traindata, TrainLabel)
print model.score(TestData, TestLabel)
# TrainData = ShuffleDataSet(TrainData)
xxx = linear_model.LogisticRegression(penalty="l2")
xxx.fit(traindata, TrainLabel)
print xxx.score(TestData, TestLabel)







