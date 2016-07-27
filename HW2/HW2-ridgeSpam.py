from numpy import *
import random 
import matplotlib.pyplot as plt


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
	for i in xrange(1,featureNum):
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
		Label.append(data[-1])
		Data.append(data[:-1])
	# Data = DataSet[:-1]
	# Label = DataSet[-1]
	# print len(Label)
	return Data, Label

def regression(arrdata,arrlabel,k):
	arrdata   =  mat(arrdata)
	arrlabel  =  mat(arrlabel)
	xTx = arrdata.T*arrdata + k * identity(58)
	# print shape(xTx) 
	w = xTx.I * arrdata.T * arrlabel.T
	return w

def Err(Pred, Tru, x):
	err = 0.0
	TP = 0
	TN = 0
	FP = 0
	FN = 0
	for i in range(len(Pred)):
		if Pred[i] > (0.0005*x):
			Pred[i] = 1
		else :
			Pred[i] = 0
	for i in range(len(Pred)):	
		if Pred[i] == 1 and Tru[i] == 1:
			err +=1
			TP  +=1
		if Pred[i] == 1 and Tru[i] == 0:
			FP += 1
		if Pred[i] == 0 and Tru[i] == 0:
			TN  += 1
			err += 1
		if Pred[i] == 0 and Tru[i] == 1:
			FN += 1
	print 'TP = %d FP = %d TN = %d FN = %d' %(TP, FP, TN, FN)
	return err/len(Pred), TP, FP, TN, FN

def ShuffleDataSet(DataSet):
	for data in DataSet:
		data.insert(0,1)
	random.shuffle(DataSet)
	return DataSet

DataSet = loadData('/Users/skylatitude/Desktop/spambase/spambase.data')
DataSet = ShuffleDataSet(DataSet)
DataSet = NormalData(DataSet)
SetNum = (len(DataSet) - 1)/10
FPR = []
TPR = []
AUC = []
for x in range(2000):
	TPos = 0.0
	FPos = 0.0
	TNeg = 0.0
	FNeg = 0.0
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
		w = regression(trainData,trainLabel,0.22)
		# print w
		predictLabel = testData * w
		MSE , TP, FP, TN, FN = Err(predictLabel, testLabel,x)
		TPos += TP
		FPos += FP
		TNeg += TN
		FNeg += FN
		

		print 'The ACC of %d Folder is %f' %(i+1,MSE)

	TPR.append(TPos/(TPos + FNeg))
	FPR.append(FPos/(FPos + TNeg))

for i in range(2000):
	plt.plot(FPR[i], TPR[i], 'b*')
	plt.plot(FPR, TPR)
a = sum(FPR)
b = sum(TPR)
AUC = -trapz(TPR,FPR)
print a , b
print AUC
plt.title('curve')
plt.show()




