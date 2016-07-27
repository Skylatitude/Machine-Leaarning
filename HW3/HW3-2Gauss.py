from numpy import *
import random
import scipy.stats as stats
import matplotlib.pyplot as plt

def loadData(filename):
	vector = []
	with open(filename) as file:
		for line in file:
			vector.append(map(float,line.split()))
	return vector

def ShuffleData(DataSet):
	
	random.shuffle(DataSet)
	return DataSet

def Normal(DataSet):

	meanvalue = []
	variance  = []
	vari1 = 0.0
	vari0 = 0.0
	a_1 = 0.0
	a_0 = 0.0
	for data in DataSet:
		if data[-1] == 1:
			a_1 += 1
		else:
			a_0 += 1
	a_1 = a_1 / len(DataSet)
	a_0 = a_0 / len(DataSet)
	DataNum = len(DataSet)
	featureNum = len(DataSet[0]) - 1
	for i in range(featureNum):
		FeatureList1 = []
		FeatureList0 = []
		FeatureList  = []
		for data in DataSet:
			FeatureList.append(data[i])
			if data[-1] == 1:
				FeatureList1.append(data[i])
			else:
				FeatureList0.append(data[i])
		mv0 = mean(FeatureList0)
		mv1 = mean(FeatureList1)
		# print len(FeatureList1), len(FeatureList0)
		# for j in range(len(FeatureList1)):
		# 	vari1 += (FeatureList1[j] - mv1) ** 2
		# vari1 = vari1/len(FeatureList1)
		# for j in range(len(FeatureList0)):
		# 	vari0 += (FeatureList0[j] - mv0) ** 2
		# vari0 = vari0/len(FeatureList0)
		vari0 = var(FeatureList0)
		vari1 = var(FeatureList1)
		vari  = var(FeatureList)
		meanvalue.append([mv0,mv1])
		variance.append([vari0,vari1,vari])
	# print variance
	return meanvalue, variance, a_0, a_1

def test(DataSet, meanvalue, variance, a_0, a_1):
	predict = []
	acc = 0.0
	for data in DataSet:
		p_1 = 1
		p_0 = 1
		for i in range(len(DataSet[0])-1):
			if variance[i][0] == 0.0:
				variance[i][0] = 0.00001
			if variance[i][1] == 0.0:
				variance[i][1] = 0.00001
			p_0 = p_0 * stats.norm.pdf(data[i], meanvalue[i][0], variance[i][0]) * a_1
			p_1 = p_1 * stats.norm.pdf(data[i], meanvalue[i][1], variance[i][1]) * a_0
		if p_1 >= p_0:
			predict.append(1)
		else :
			predict.append(0)
		if predict[-1] == data[-1]:
			acc += 1
	return acc / len(DataSet)

def test2(DataSet, meanvalue, variance, a_0, a_1):
	lnp = []
	for data in DataSet:
		p_1 = 1
		p_0 = 1
		for i in range(len(DataSet[0])-1):
			if variance[i][0] == 0.0:
				variance[i][0] = 0.00001
			if variance[i][1] == 0.0:
				variance[i][1] = 0.00001
			p_0 = p_0 * stats.norm.pdf(data[i], meanvalue[i][0], variance[i][0]) * a_1
			p_1 = p_1 * stats.norm.pdf(data[i], meanvalue[i][1], variance[i][1]) * a_0
		if p_1 == 0 or p_0 ==0:
			p_1 = p_1+10**(-100)
			p_0 = p_0+10**(-100)
		p = math.log((p_1/p_0),10)
		lnp.append(p)
	return lnp

def pic(DataSet, lnp, t):
	predict = []
	acc = 0.0
	TP = 0
	TN = 0
	FP = 0
	FN = 0
	for i in range(len(DataSet)):
		if lnp[i] > t:
			predict.append(1)

		else :
			predict.append(0)
		if predict[i] == 1 and DataSet[i][-1] == 1:
			TP += 1
			acc += 1
		if predict[i] == 1 and DataSet[i][-1] == 0:
			FP += 1
		if predict[i] == 0 and DataSet[i][-1] == 0:
			TN += 1
			acc += 1
		if predict[i] == 0 and DataSet[i][-1] == 1:
			FN += 1
	print 'TP = %d FP = %d TN = %d FN = %d' %(TP, FP, TN, FN)
	return acc / len(DataSet), TP, FP, TN, FN

def roc(DataSet, SetNum):
	FPR = []
	TPR = []
	

	Tsdata = DataSet[-SetNum:]
	Trdata = DataSet[0:-SetNum]
	meanvalue, result, a_0, a_1 = Normal(Trdata)
	lnp = test2(Tsdata,meanvalue, result, a_0, a_1)
	t = sorted(set(lnp))
	print len(t)
	for x in range(len(t)):		
		TPos = 0.0
		FPos = 0.0
		TNeg = 0.0
		FNeg = 0.0
		ACC, TP, FP, TN, FN = pic(Tsdata, lnp, t[x])
		TPos += TP
		FPos += FP 
		TNeg += TN 
		FNeg += FN
		TPR.append(TPos/(TPos + FNeg))
		FPR.append(FPos/(FPos + TNeg))
	for i in range(len(t)):
		plt.plot(FPR[i], TPR[i], 'b*')
		plt.plot(FPR, TPR)
	a = sum(FPR)
	b = sum(TPR)
	AUC = -trapz(TPR,FPR)
	print AUC
	plt.title('Normal Curve')
	plt.show()


DataSet = loadData('/Users/skylatitude/Desktop/spambase/spambase.data')
DataSet = ShuffleData(DataSet)
SetNum = (len(DataSet) -1 ) / 10
for i in range(10):
	TestData = DataSet[SetNum*i:(SetNum*i + SetNum)]
	TrainData = DataSet[0: SetNum*i]
	TrainData.extend(DataSet[(SetNum*i + SetNum):-1])
	meanvalue, result, a_0, a_1 = Normal(TrainData)
	acc = test(TestData,meanvalue, result, a_0, a_1)
	print 'The ACC of %d Folder is %f' %(i+1,acc)
roc(DataSet, SetNum)

