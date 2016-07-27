from numpy import *
import random
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

def Prob(DataSet):


	result = []
	meanvalue = []
	Featurelist = []
	DataNum = len(DataSet)
	featureNum = len(DataSet[0]) - 1
	for i in range(featureNum):	
		a_0 = 0.0
		a_1 = 0.0
		p1 = 0.0 #<=mean value/spam
		p2 = 0.0 # >mean value/spam
		p3 = 0.0 #<=mean value/non-spam
		p4 = 0.0 # >mean value/non-spam
		for data in DataSet:
			Featurelist.append(data[i])
		mv = mean(Featurelist)
		meanvalue.append(mv)
		for data in DataSet:
			if data[-1] == 1 and data[i] <= mv:
				p1 += 1
				a_1 += 1
			elif data[-1] == 1 and data[i] > mv:
				p2 += 1
				a_1 += 1
			elif data[-1] == 0 and data[i] <= mv:
				p3 += 1
				a_0 += 1
			elif data[-1] == 0 and data[i] > mv:
				p4 +=1
				a_0 += 1
		p1 = p1 / DataNum
		p2 = p2 / DataNum
		p3 = p3 / DataNum
		p4 = p4 / DataNum
		result.append([p1,p2,p3,p4])
	return meanvalue, result, a_0/DataNum, a_1/DataNum

def test(DataSet, meanvalue, result, a_0, a_1):
	predict = []
	acc =0.0
	for data in DataSet:
		p_1 = 1
		p_0 = 1
		for i in range(len(DataSet[0])-1):
			if data[i] <= meanvalue[i]:
				p_1 = p_1 * result[i][0] * a_0
				p_0 = p_0 * result[i][2] * a_1
			else:
				p_1 = result[i][1] * p_1 * a_0
				p_0 = result[i][3] * p_0 * a_1
		if p_1 >= p_0:
			predict.append(1)

		else :
			predict.append(0)
		if data[-1] == predict[-1]:
			acc += 1
	return acc / len(DataSet)

def test2(DataSet, meanvalue, result, a_0, a_1):

	lnp = []
	for data in DataSet:
		p_1 = 1
		p_0 = 1
		for i in range(len(DataSet[0])-1):
			if data[i] <= meanvalue[i]:
				p_1 = p_1 * result[i][0] * a_0
				p_0 = p_0 * result[i][2] * a_1
			else:
				p_1 = result[i][1] * p_1 * a_0
				p_0 = result[i][3] * p_0 * a_1
		if p_1 == 0 or p_0 ==0:
			p_1 = p_1+10**(-100)
			p_0 = p_0+10**(-100)
		p = math.log((p_1/p_0),10)
		lnp.append(p)
	return lnp
def pic(DataSet, lnp, x):
	predict = []
	acc =0.0
	TP = 0
	TN = 0
	FP = 0
	FN = 0
	for i in range(len(DataSet)):
		if lnp[i] > min(lnp) + x*((max(lnp) - min(lnp) + 10)/200):
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
	meanvalue, result, a_0, a_1 = Prob(Trdata)
	lnp = test2(Tsdata, meanvalue, result, a_0, a_1)
	for x in range(200):
		TPos = 0.0
		FPos = 0.0
		TNeg = 0.0
		FNeg = 0.0
		ACC, TP, FP, TN, FN = pic(Tsdata,lnp,x)
		TPos += TP
		FPos += FP 
		TNeg += TN 
		FNeg += FN
		TPR.append(TPos/(TPos + FNeg))
		FPR.append(FPos/(FPos + TNeg))
	for i in range(200):
		plt.plot(FPR[i], TPR[i], 'b*')
		plt.plot(FPR, TPR)
	a = sum(FPR)
	b = sum(TPR)
	AUC = -trapz(TPR,FPR)
	print AUC
	plt.title('Bernoulli Curve')
	plt.show()

DataSet = loadData('spambase.data')
DataSet = ShuffleData(DataSet)
SetNum = (len(DataSet) -1 ) / 10
for i in range(10):
	TestData = DataSet[SetNum*i:(SetNum*i + SetNum)]
	TrainData = DataSet[0: SetNum*i]
	TrainData.extend(DataSet[(SetNum*i + SetNum):])
	meanvalue, result, a_0, a_1 = Prob(TrainData)
	acc = test(TestData,meanvalue, result, a_0, a_1)
	print 'The ACC of %d Folder is %f' %(i+1,acc)
roc(DataSet, SetNum)


