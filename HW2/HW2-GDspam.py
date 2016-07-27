from numpy import *
import random 
from sklearn import metrics
import matplotlib.pyplot as plt

def loadData(filename):
	vector = []
	with open(filename) as file:
		for line in file:
			vector.append(map(float,line.split()))
	return vector

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
		
		Label.append(data[-1])
		Data.append(data[:-1])
	# Data = DataSet[:-1]
	# Label = DataSet[-1]
	# print len(Label)
	return Data, Label

def ShuffleDataSet(DataSet):
	random.shuffle(DataSet)
	return DataSet

# def optimalW(X, Y, Lambda):
#     Xt = transpose(X)
#     m, n = shape(X)
#     W = ones((n, 1))
#     print shape(X), shape(Y)
#     Wr = [0] * n
#     for a in range(3):
#         for i in range(n):
#             for j in range(m):
#                 Hw = 1.0/(1 + exp(- (X[j] * W)))
#                 W[i, 0] += Lambda * (Y[j] - Hw) * X[j, i]
#     return transpose(W)
def GradientDescent(traindata, trainlabel, k):
	# traindata = mat(traindata)
	# trainlabel = mat(trainlabel)
	m , n = shape(traindata)
	w = ones((n,1))
	# weight = []
	while(1):
		for i in range(n):
			for j in range(m):
				h = dot(traindata[j],w) - trainlabel[j] 
				w[i] = w[i] - h * k * traindata[j][i]
		predictLabel = dot(traindata,w)
		acc = Err(predictLabel,trainLabel)
		if acc > 0.7:
			break
	return w

def Err(Pred, Tru):
	err = 40.0
	for i in range(len(Pred)):
		if Pred[i] >= 0.5:
			Pred[i] = 1
		else:
			Pred[i] = 0
	# print Pred, Tru
	# print Pred[0] == Tru[0]
	for i in range(len(Pred)):
		if Pred[i] == Tru[i]:
			err = err +1
	# print err
	return err/len(Pred)

DataSet = loadData('spambase.data')
DataSet = NormalData(DataSet)
DataSet = ShuffleDataSet(DataSet)
SetNum = (len(DataSet) - 1)/10
for i in range(1):
# i = 5
	TestData = DataSet[SetNum*i:(SetNum*i+SetNum)]
	# Label = GetFeature(TestData,-1)
	TrainData = DataSet[0:SetNum*i]
	TrainData.extend(DataSet[(SetNum*i+SetNum):-1])
# TestSet = loadData('/Users/skylatitude/Desktop/test set.txt')

	trainData, trainLabel = SplitData(TrainData)
	testData, testLabel   = SplitData(TestData)
# testData, testLabel   = SplitData(TestSet)
	w = GradientDescent(trainData,trainLabel, 0.1)
	# print 'The %d w is %d:' %(i+1,len(w))
	predictLabel = dot(testData,w)
	# ACC = Err(predictLabel, testLabel)
	# print len(predictLabel)
	# print len(testLabel)
	
	# print 'The ACC of %d Folder is %f' %(i+1,ACC)
fpr, tpr, thresholds = metrics.roc_curve(predictLabel, testLabel, pos_label=1)
for i in range(len(fpr)):
    plt.plot(fpr[i], tpr[i], "b*")
    plt.plot(fpr, tpr)
plt.title("KNN")
plt.show()
print "AUC : ", metrics.auc(fpr, tpr)
print thresholds


