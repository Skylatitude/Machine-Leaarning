from numpy import *
import numpy
from sklearn import metrics
import matplotlib.pyplot as plt

def loadData(filename):
	vector = []
	with open(filename) as file:

		for line in file:
			vector.append(map(float,line.split(" ")))
	return vector

def ShuffleData(DataSet):
	
	random.shuffle(DataSet)
	return DataSet

def Gaussian(DataSet):
	onenum = 0.0
	zeronum = 0.0
	featureone = array([0.0] * (len(DataSet[0])-1))
	featurezreo = array([0.0] * (len(DataSet[0])-1))
	for data in DataSet:
		if data[-1] == 1:
			featureone += data[:-1]
			onenum += 1
		else:
			featurezreo += data[:-1]
			zeronum += 1
	mv1 = true_divide(featureone, onenum)
	mv0 = true_divide(featurezreo, zeronum)
	p1 = onenum / len(DataSet)
	mitr = array([[0 for i in range(len(DataSet[0])-1)] for j in range(len(DataSet[0])-1)])
	for data in DataSet:
		if data[-1] == 1:
			mitr += dot(transpose([(data[:-1] - mv1)]), [(data[:-1] - mv1)])
		else:
			mitr += dot(transpose([(data[:-1] - mv0)]), [(data[:-1] - mv0)])
	mitr = true_divide(mitr, len(DataSet))
	return mv0, mv1, mitr, p1

def test(mv0, mv1, mitr, p, DataSet):
	predict = []
	acc = 0.0
	k = 1 / ((math.pi * 2) ** ((len(DataSet[0])-1.0)/2.0))
	for data in DataSet:
		p0 = k * exp(dot(dot(([(data[:-1] - mv0)]), linalg.pinv(mitr)), transpose([(data[0:-1] - mv0)])) * -0.5)
		p1 = k * exp(dot(dot(([(data[:-1] - mv1)]), linalg.pinv(mitr)), transpose([(data[0:-1] - mv1)])) * -0.5)
		if (p0 * p) < (p1 *(1 - p)):
			predict.append(1)
		else:
			predict.append(0)
		if predict[-1] == data[-1]:
			acc += 1
	fpr, tpr, thresholds = metrics.roc_curve(a, b, pos_label=1)
	for i in range(len(fpr)):
	    plt.plot(tpr[i],tpr[i], "b*")
	    plt.plot(fpr, tpr)
	plt.title("KNN")
	plt.show()
	print "AUC : ", metrics.auc(fpr, tpr)
	return acc/ len(DataSet)

DataSet = loadData('spambase.data')
DataSet = ShuffleData(DataSet)
SetNum = (len(DataSet) -1 ) / 10
for i in range(1):
	TestData = DataSet[SetNum*i:(SetNum*i + SetNum)]
	TrainData = DataSet[0: SetNum*i]
	TrainData.extend(DataSet[(SetNum*i + SetNum):-1])
	mv0, mv1, mitr, p1 = Gaussian(TrainData)
	acc = test(mv0, mv1, mitr, p1, DataSet)
	print 'The ACC of %d Folder is %f' %(i+1,acc)

