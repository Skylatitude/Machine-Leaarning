from numpy import *
import time

def loadData(filename):
	vector = []
	with open(filename) as file:
		for line in file:
			vector.append(map(float,line.split()))
	return vector

def GetFeature(DataSet, num):
	Featurelist = []
	for vector in DataSet:
		Featurelist.append(vector[num])
	return Featurelist

def CalcuVar(DataSet):
	numData = len(DataSet)
	if numData == 0:
		return 0
	var = []
	Valuelist = GetFeature(DataSet, -1)
	# print Valuelist
	average = float(sum(Valuelist))/numData
	# print average
	for i in range(numData):
		var.append((Valuelist[i] - average)**2)
	variance = sum(var)
	return variance
	# return var(Valuelist)

def MeanVal(DataSet):
	Valuelist = GetFeature(DataSet,-1)

	return float(sum(Valuelist))/len(Valuelist)

def SplitDataSet(DataSet, feature, val):
	leftDataSet = []
	rightDataSet = []
	for vector in DataSet:
		if vector[feature] <= val:
			leftDataSet.append(vector)
		else :
			rightDataSet.append(vector)
	return leftDataSet, rightDataSet

def TheBestChoice(DataSet):
	OrienVar    = CalcuVar(DataSet)
	bestVar     = inf
	deff        = 0.0
	bestval     = 0.0
	bestfeature = -1
	FeatureNum   = len(DataSet[0]) - 1
	bestLeft    = []
	bestright   = []
	for i in range(FeatureNum):
		Featurelist = GetFeature(DataSet,i)
		ShortFeaturelist  =  list(set(Featurelist))
		# print len(ShortFeaturelist)
		for n in range(len(ShortFeaturelist)):
			leftDataSet, rightDataSet = SplitDataSet(DataSet, i, ShortFeaturelist[n])
			leftVar = CalcuVar(leftDataSet)
			rightVar = CalcuVar(rightDataSet)
			# print len(leftDataSet)+len(rightDataSet)
			if leftVar + rightVar <= bestVar:
				bestVar = leftVar+rightVar
				bestfeature = i
				bestval     = ShortFeaturelist[n]
				bestLeft    = leftDataSet
				bestright   = rightDataSet
				# print bestVar
	deff = OrienVar - bestVar
	return bestfeature, bestval, bestLeft, bestright, deff

def CreatTree(DataSet):
	# trS = ops[0]
	# trN = ops[1]
	# if len(DataSet) <= trN:
	# 	# print GetFeature(DataSet,-1)
	# 	return {'End': MeanVal(DataSet)}
	bestfeature, bestval, bestLeft, bestright, deff = TheBestChoice(DataSet)
	regTree = {}
	regTree['vertex'] = bestfeature
	regTree['leaf']  = bestval
	regTree['lefttree'] = MeanVal(bestLeft)
	regTree['righttree'] = MeanVal(bestright)
	return regTree

def PredictValue(TestData, regTree):
	Predictlist = []
	temptree = regTree
	for i in range(len(TestData)):
		if TestData[i][temptree['vertex']] <= temptree['leaf']:
			Predictlist.append(temptree['lefttree'])
		if TestData[i][temptree['vertex']] > temptree['leaf']:
			Predictlist.append(temptree['righttree'])
	return Predictlist

def PredictTest(Testdata, regTree):
	Predictlist = []
	lefttest, righttest = SplitDataSet(Testdata, regTree['vertex'], regTree['leaf'])
	for i in range(len(Testdata)):
		if Testdata[i][regTree['vertex']] <= regTree['leaf']:
			Predictlist.append(regTree['lefttree'])
		else:

			Predictlist.append(regTree['righttree'])
	return Predictlist

def regErr(Predictlist,Label):
	Err = 0.0
	for i in range(len(Predictlist)):
		Err += (Predictlist[i]-Label[i])**2
	Err = Err/len(Label)
	return Err




DataSet = loadData('/Users/skylatitude/Desktop/train set.txt')
TestSet = loadData('/Users/skylatitude/Desktop/test set.txt')
Predictlist = []
result = []
Label = GetFeature(DataSet,-1)
Testlabel = GetFeature(TestSet, -1)
predict = zeros((len(DataSet), 1))
testpredict = zeros((len(TestSet), 1))
for i in range(10):
	Tree = CreatTree(DataSet)
	result.append(Tree)
	Predictlist = PredictValue(DataSet, Tree)
	for j in range(len(DataSet)):
		DataSet[j][-1] = DataSet[j][-1] - Predictlist[j]
		predict[j] += Predictlist[j]

# fin = GetFeature(DataSet, 0)
# print fin
# print MeanVal(fin)
# var = CalcuVar([[1,2,1],[1,2,3],[4,3,4]])
# print var
# bestfeature, bestval= TheBestChoice(DataSet)
	error = regErr(predict, Label)
	print 'The %d train variance is %f' %(i + 1 ,error)
for i in range(len(result)):
	print 'The %d test :' %(i+1)
	predictlist = PredictTest(TestSet, result[i])
	for j in range(len(predictlist)):
		TestSet[j][-1] = TestSet[j][-1] - predictlist[j]
		testpredict[j] += predictlist[j]
# print shape(testpredict), shape(Testlabel)
	testerr = regErr(testpredict, Testlabel)
	print 'The test variance is %f' %testerr


