from numpy import *
import random 


def loadData(filename):
	vector = []
	with open(filename) as file:
		for line in file:
			vector.append(map(float,line.split()))
	return vector
# '/Users/skylatitude/Desktop/train set.txt'

def NormalData(DataSet):
	# FullSet = []
	Featurelist = []
	# FullSet.extend(DataSet)
	# FullSet.extend(TestSet)
	# print len(FullSet)
	DataNum = len(DataSet)
	featureNum = len(DataSet[0]) - 1
	for i in range(featureNum):
		for data in DataSet:
			Featurelist.append(data[i])
		maxVal = max(Featurelist)
		minVal = min(Featurelist)
		RangeVal = maxVal - minVal
		for vector in DataSet:
			vector[i] = (vector[i] - minVal)/RangeVal
	# DataSet = FullSet[:DataNum]
	# TestSet = FullSet[DataNum:]
	# print len(TestSet)
	return DataSet

def SplitData(DataSet):
	Data = []
	Label = []
	for data in DataSet:
		Data.append(data[:-1])
		Label.append(data[-1])
	# Data = DataSet[:-1]
	# Label = DataSet[-1]
	# print len(Label)
	return Data, Label

def regression(arrdata,arrlabel):
	arrdata   =  mat(arrdata)
	arrlabel  =  mat(arrlabel)
	xTx = arrdata.T*arrdata
	w = xTx.I*arrdata.T*arrlabel.T
	return w

def Err(Pred, Tru):
	err = 0.0
	for i in range(len(Pred)):
		err += (Tru[i] - Pred[i]) **2
	return err/len(Pred)

def ShuffleDataSet(DataSet):
	random.shuffle(DataSet)
	return DataSet

DataSet = loadData('/Users/skylatitude/Desktop/spambase/spambase.data')
DataSet = NormalData(DataSet)
DataSet = ShuffleDataSet(DataSet)
SetNum = (len(DataSet) - 1)/10
for i in range(10):
# i = 5
	TestData = DataSet[SetNum*i:(SetNum*i+SetNum)]
	# Label = GetFeature(TestData,-1)
	TrainData = DataSet[0:SetNum*i]
	TrainData.extend(DataSet[(SetNum*i+SetNum):-1])
# TestSet = loadData('/Users/skylatitude/Desktop/test set.txt')

	trainData, trainLabel = SplitData(TrainData)
	testData, testLabel   = SplitData(TestData)
# testData, testLabel   = SplitData(TestSet)
	w = regression(trainData,trainLabel)
	predictLabel = testData * w
	MSE = Err(predictLabel, testLabel)
	print 'The MSE of %d Folder is %f' %(i+1,MSE)
