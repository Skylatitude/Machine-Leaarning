from numpy import *
from math import log
import time
import random 

def loadData(filename):
	vector = []
	with open(filename) as file:
		for line in file:
			vector.append(map(float,line.split()))
	return vector

def CalculationShannonEntropy(DataSet):
	numEntries = len(DataSet)
	labelCounts = {}
	for FeatureVector in DataSet:
		currentLabel = FeatureVector[-1]
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1
	shannonEntropy = 0.0
	for key in labelCounts:
		prob = float(labelCounts[key])/numEntries
		shannonEntropy -= prob * log(prob,2)
	# print shannonEntropy
	return shannonEntropy

def SplitDataSet(DataSet, feature, val):
	leftDataSet = []
	rightDataSet = []
	for vector in DataSet:
		if vector[feature] <= val:
			leftDataSet.append(vector)
		else :
			rightDataSet.append(vector)
	return leftDataSet, rightDataSet

def GetFeature(DataSet, num):
	Featurelist = []
	for vector in DataSet:
		Featurelist.append(vector[num])
	return Featurelist

def TheBestChoice(DataSet):
	OrienEntr   = CalculationShannonEntropy(DataSet)
	bestEntr    = inf
	IG          = 0.0
	bestval     = 0.0
	bestfeature = -1
	FeatureNum  = len(DataSet[0]) - 1
	bestLeft    = []
	bestright   = []
	for i in range(FeatureNum):
		Featurelist = GetFeature(DataSet, i)
		ShortFeaturelist  =  list(set(Featurelist))
		# print ShortFeaturelist
		for n in range(len(ShortFeaturelist)):
			leftDataSet, rightDataSet = SplitDataSet(DataSet, i, ShortFeaturelist[n])
			Probleft = float(len(leftDataSet))/len(DataSet)
			Probright = float(len(rightDataSet))/len(DataSet)
			leftEntro = Probleft * CalculationShannonEntropy(leftDataSet)
			rightEntro = Probright * CalculationShannonEntropy(rightDataSet)
			if leftEntro + rightEntro <= bestEntr:
				bestEntr = leftEntro + rightEntro
				bestfeature = i
				bestval     = ShortFeaturelist[n]
				bestLeft    = leftDataSet
				bestright   = rightDataSet
	IG  =   OrienEntr - bestEntr
	return bestfeature, bestval, IG, bestLeft, bestright

def CreatTree(DataSet):
	feature, val, IG, leftDataSet, rightDataSet = TheBestChoice(DataSet)
	if IG == 0.0:
		return {'End': DataSet[0][-1]}
	desTree = {}
	desTree['vertex'] = feature
	desTree['leaf']   = val
	desTree['lefttree']  = CreatTree(leftDataSet)
	desTree['righttree'] = CreatTree(rightDataSet)
	return desTree
	
def PredictValue(TestData, desTree):
	Predictlist = []
	for i in range(len(TestData)):
		temptree = desTree
		while('End' not in temptree.keys()):
			if TestData[i][temptree['vertex']] <= temptree['leaf']:
				temptree = temptree['lefttree']
				continue
			if TestData[i][temptree['vertex']] > temptree['leaf']:
				temptree = temptree['righttree']
		Predictlist.extend(temptree.values())
	return Predictlist

def ShuffleDataSet(DataSet):
	random.shuffle(DataSet)
	return DataSet

def DesErr(Predictlist, Label):
	rate = 0.0
	for i in range(len(Predictlist)):
		if Predictlist[i] == Label[i]:
			rate += 1
	rate = rate/len(Label)
	return rate


DataSet = loadData('/Users/skylatitude/Desktop/spambase/spambase.data')
DataSet = ShuffleDataSet(DataSet)
# print type(DataSet)
TestData = []
TrainData = []
Predictlist = []
Label = []
SetNum = (len(DataSet) - 1)/10
# print SetNum
for i in range(10):
# i = 5
	TestData = DataSet[SetNum*i:(SetNum*i+SetNum)]
	Label = GetFeature(TestData,-1)
	TrainData = DataSet[0:SetNum*i]
	TrainData.extend(DataSet[(SetNum*i+SetNum):-1])
	# print Label
	# print len(TrainData)
	start = time.clock()
# entropy = CalculationShannonEntropy(DataSet)
# leftDataSet, rightDataSet = SplitDataSet(DataSet, 0, 1.0)
# Featurelist = GetFeature(DataSet, 1)
# feature, val, IG, leftDataSet, rightDataSet = TheBestChoice(DataSet)
# print IG
# print DataSet[-1][-1]
	Tree = CreatTree(TrainData)
	Predictlist = PredictValue(TestData, Tree)
	Rate = DesErr(Predictlist, Label)
	print'The accuracy of %d Folder is : %f' %(i+1, Rate)
	# print Predictlist
# print Tree

	finish = time.clock()
	print 'The %d Folder\'s running time is : %f' %(i+1,finish - start)




