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
	prob1 = sum([vector[-1] for vector in DataSet])
	prob0 = numEntries - prob1
	shannonEntropy = 0.0
	if prob0 != 0.0:
		prob = float(prob0) / numEntries
		shannonEntropy -= prob * log(prob, 2)
	if prob1 != 0.0:
		prob = float(prob1)/ numEntries
		shannonEntropy -= prob * log(prob, 2)
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

# def CreatTree(DataSet):
# 	feature, val, IG, leftDataSet, rightDataSet= TheBestChoice(DataSet)
# 	if IG == 0.0:
# 		return {'End': DataSet[0][-1]}
# 	desTree = {}
# 	desTree['vertex'] = feature
# 	desTree['leaf']   = val
# 	desTree['lefttree']  = CreatTree(leftDataSet)
# 	desTree['righttree'] = CreatTree(rightDataSet)
# 	return desTree
	
# def PredictValue(TestData, desTree):
# 	Predictlist = []
# 	for i in range(len(TestData)):
# 		temptree = desTree
# 		while('End' not in temptree.keys()):
# 			if TestData[i][temptree['vertex']] <= temptree['leaf']:
# 				temptree = temptree['lefttree']
# 				continue
# 			if TestData[i][temptree['vertex']] > temptree['leaf']:
# 				temptree = temptree['righttree']
# 		Predictlist.extend(temptree.values())
# 	return Predictlist

def ShuffleDataSet(DataSet):
	random.shuffle(DataSet)
	return DataSet

def DesErr(Predictlist, Label):
	rate = 0.0
	print Predictlist
	for i in range(len(Label)):
		if Predictlist[0,i] <= 0.5 and Label[i] == 0:
			rate += 1
		elif Predictlist[0,i] > 0.5 and Label[i] == 1:
			rate += 1
	rate = rate/len(Label)
	return rate

def GetLabels(DataSet):
	one = 0
	zero = 0
	for i in range(len(DataSet)):
		if DataSet[-1] == 1:
			one += 1
		else:
			zero += 1
	if one >= zero:
		return 1
	else:
		return 0

def CreatTree(DataSet, ops = 200):
	# if len(DataSet) <= ops:
	# 	return {'End': GetLabels(DataSet)}
	# print 'ddddddddddddd'
	feature, val, IG, leftDataSet, rightDataSet= TheBestChoice(DataSet)
	# print shape(leftDataSet)
	if IG == 0.0:
		return {'End': DataSet[0][-1]}
	desTree = {}
	desTree['vertex'] = feature
	desTree['leaf']   = val
	desTree['lefttree']  = CreatTree(leftDataSet)
	desTree['righttree'] = CreatTree(rightDataSet)
	return desTree

def randomdata(DataSet):
	m, n = shape(DataSet)
	train = []
	trainset = []
	for i in range(m):
		train.append(random.randint(0,m-1))
	train = list(set(train))
	for i in range(len(train)):
		trainset.append(DataSet[train[i]])
	return trainset

def bagging(DataSet):
	result = []

	for i in range(50):
		traindata = randomdata(DataSet)
		print shape(traindata)
		tree = CreatTree(traindata)
		print 'The %d training.... ' %(i+1)
		result.append(tree)
	return result

def baggingTest(TestData, result):
	m = len(TestData)
	predictlist = mat(zeros((1,m)))
	for i in range(len(result)):
		predictlist += mat(PredictValue(TestData, result[i]))
	predictlist = predictlist / float(len(result))
	return predictlist
	
def PredictValue(TestData, desTree):
	Predictlist = []
	for i in range(len(TestData)):
		temptree = desTree
		while('End' not in temptree.keys()):
			# if 'End' in temptree.keys():
				# Predictlist.append(temptree['End'])
				# break
			if TestData[i][temptree['vertex']] <= temptree['leaf']:
				temptree = temptree['lefttree']
				continue
			if TestData[i][temptree['vertex']] > temptree['leaf']:
				temptree = temptree['righttree']
			
		# Predictlist.append(temptree.values())
	print shape([Predictlist])
	return Predictlist


DataSet = loadData('/Users/skylatitude/Desktop/spambase/spambase.data')
DataSet = ShuffleDataSet(DataSet)
# print type(DataSet)
TestData = []
TrainData = []
Predictlist = []

SetNum = (len(DataSet) - 1)/10
# print SetNum

i = 0
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
# Tree = CreatTree(TrainData)
# Predictlist = PredictValue(TestData, Tree)
result = bagging(TrainData)
Predictlist = baggingTest(TestData, result)
# print Predictlist,shape(Predictlist)
# print Label, shape(Label)
Rate = DesErr(Predictlist, Label)
print'The accuracy of %d Folder is : %f' %(i+1, Rate)
# print Predictlist
# print Tree

finish = time.clock()
print 'The %d Folder\'s running time is : %f' %(i+1,finish - start)
