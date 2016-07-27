from numpy import *

def loadData(filename):
	vector = []
	with open(filename) as file:
		for line in file:
			vector.append(map(float,line.split()))
	return vector
# '/Users/skylatitude/Desktop/train set.txt'

def NormalData(DataSet, TestSet):
	FullSet = []
	Featurelist = []
	FullSet.extend(DataSet)
	FullSet.extend(TestSet)
	print len(FullSet)
	DataNum = len(DataSet)
	featureNum = len(FullSet[0]) - 1
	for i in range(featureNum):
		for data in FullSet:
			Featurelist.append(data[i])
		maxVal = max(Featurelist)
		minVal = min(Featurelist)
		RangeVal = maxVal - minVal
		for vector in FullSet:
			vector[i] = (vector[i] - minVal)/RangeVal
	DataSet = FullSet[:DataNum]
	TestSet = FullSet[DataNum:]
	# print len(TestSet)
	return DataSet, TestSet

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

DataSet = loadData('/Users/skylatitude/Desktop/train set.txt')
TestSet = loadData('/Users/skylatitude/Desktop/test set.txt')
DataSet, TestSet = NormalData(DataSet, TestSet)
trainData, trainLabel = SplitData(DataSet)
testData, testLabel   = SplitData(TestSet)
w = regression(trainData,trainLabel)
predictLabel = testData * w
MSE = Err(predictLabel, testLabel)
print 'The MSE is %f' %MSE
