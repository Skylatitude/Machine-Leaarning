from numpy import *
import random
import math
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

# def his(DataSet):
# 	meanvalue = []
# 	DataNum = len(DataSet)
# 	featureNum = len(DataSet[0]) - 1
# 	for i in range(featureNum):
# 		featurelist1 = []
# 		featurelist0 = []
# 		left0 = []
# 		left1 = []
# 		right0 = []
# 		right1 = []
# 		# y = 0
# 		p1 = 0.0
# 		p2 = 0.0
# 		p3 = 0.0
# 		p4 = 0.0
# 		# y = 1
# 		p5 = 0.0
# 		p6 = 0.0
# 		p7 = 0.0
# 		p8 = 0.0
# 		for data in DataSet:
# 			if data[-1] == 1:
# 				featurelist1.append(data[i])
# 			else :
# 				featurelist0.append(data[i])
# 		mv1 = mean(featurelist1)
# 		mv0 = mean(featurelist0)
# 		for j in range(len(featurelist0)):
# 			if featurelist0[j] <= mv0:
# 				left0.append(featurelist0[j])
# 			else :
# 				right0.append(featurelist0[j])
# 		for j in range(len(featurelist1)):
# 			if featurelist1[j] <= mv1:
# 				left1.append(featurelist1[j])
# 			else :
# 				right1.append(featurelist1[j])
# 		for j in range(left0):
# 			if left0[j] <= leftmv0:
# 				p1 += 1
# 			else:
# 				p2 += 1
# 		for j in range(right0):
# 			if right0[j]
# 		leftmv1 = mean(left1)
# 		leftmv0 = mean(left0)
# 		rightmv1 = mean(right1)
# 		rightmv0 = mean(right0)
# 		meanvalue.append([leftmv0,mv0,rightmv0,leftmv1,mv1,rightmv1])

def ave(data):
	mv = mean(data)
	left, right = split(data, mv)
	return mv, left, right

def split(data, mv):
	left = []
	right = []
	for i in range(len(data)):
		if data[i] <= mv:
			left.append(data[i])
		else :
			right.append(data[i])
	return left, right 


def getfeature(DataSet, i):
	FeatureList = []
	DataNum = len(DataSet)
	featureNum = len(DataSet[0]) - 1
	for data in DataSet:
		FeatureList.append(data[i])
	return FeatureList

def his(DataSet):
	result = []
	meanvalue = []
	DataNum = len(DataSet)
	featureNum = len(DataSet[0]) - 1
	for i in range(featureNum):
		a_0 = 0.0
		a_1 = 0.0
		# y = 0
		p1 = 0.0
		p2 = 0.0
		p3 = 0.0
		p4 = 0.0
		# y = 1
		p5 = 0.0
		p6 = 0.0
		p7 = 0.0
		p8 = 0.0
		Featurelist = getfeature(DataSet, i)
		featurelistleft = []
		featurelistright = []
		mv = mean(Featurelist)
		for data in DataSet:
			if data[i] <= mv:
				featurelistleft.append(data[i])
			else :
				featurelistright.append(data[i])
		leftmv = mean(featurelistleft)
		rightmv = mean(featurelistright)
		for data in DataSet:
			if data[-1] == 0:
				if data[i] <= leftmv:
					p1 += 1
				elif data[i] <= mv:
					p2 += 1
				elif data[i] <= rightmv:
					p3 += 1
				else :
					p4 += 1
				a_0 += 1
			if data[-1] == 1:
				if data[i] <= leftmv:
					p5 += 1
				elif data[i] <= mv:
					p6 += 1
				elif data[i] <= rightmv:
					p7 += 1
				else :
					p8 += 1
				a_1 += 1

		meanvalue.append([leftmv, mv, rightmv])
		p1 = p1 / DataNum
		p2 = p2 / DataNum
		p3 = p3 / DataNum
		p4 = p4 / DataNum
		p5 = p5 / DataNum
		p6 = p6 / DataNum
		p7 = p7 / DataNum
		p8 = p8 / DataNum
		result.append([p1,p2,p3,p4,p5,p6,p7,p8])
	return meanvalue, result, a_0 / DataNum, a_1 / DataNum


def test(DataSet, meanvalue, result, a_0, a_1):
	predict = []
	acc =0.0
	for data in DataSet:
		p_1 = 1
		p_0 = 1
		for i in range(len(DataSet[0])-1):
			if data[i] <= meanvalue[i][0]:
				p_1 = p_1 * result[i][4] * a_0
				p_0 = p_0 * result[i][0] * a_1
			elif data[i] <= meanvalue[i][1]:
				p_1 = result[i][5] * p_1 * a_0
				p_0 = result[i][1] * p_0 * a_1
			elif data[i] <= meanvalue[i][2]:
				p_1 = result[i][6] * p_1 * a_0
				p_0 = result[i][2] * p_0 * a_1
			else :
				p_1 = result[i][7] * p_1 * a_0
				p_0 = result[i][3] * p_0 * a_1
		if p_1 >= p_0:
			predict.append(1)

		else :
			predict.append(0)

		if data[-1] == predict[-1]:
			acc += 1
	return acc / len(DataSet)

def test2(DataSet, meanvalue, result, a_0, a_1, x):
	predict = []
	acc =0.0
	TP = 0
	TN = 0
	FP = 0
	FN = 0
	lnp = []
	spam = 0.0
	for data in DataSet:
		spam += data[-1]
	nonspam = len(DataSet) - spam

	for data in DataSet:
		p_1 = 1
		p_0 = 1
		for i in range(len(DataSet[0])-1):
			if data[i] <= meanvalue[i][0]:
				p_1 = p_1 * result[i][4] * a_0
				p_0 = p_0 * result[i][0] * a_1
			elif data[i] <= meanvalue[i][1]:
				p_1 = result[i][5] * p_1 * a_0
				p_0 = result[i][1] * p_0 * a_1
			elif data[i] <= meanvalue[i][2]:
				p_1 = result[i][6] * p_1 * a_0
				p_0 = result[i][2] * p_0 * a_1
			else :
				p_1 = result[i][7] * p_1 * a_0
				p_0 = result[i][3] * p_0 * a_1

		if p_1 == 0 or p_0 ==0:
			p_1 = p_1+10**(-100)
			p_0 = p_0+10**(-100)
		p = math.log((p_1/p_0),10)
		lnp.append(p)
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
	print max(lnp), min(lnp)
	return acc / len(DataSet), TP, FP, TN, FN

def roc(DataSet, SetNum):
	FPR = []
	TPR = []
	for x in range(200):
		TPos = 0.0
		FPos = 0.0
		TNeg = 0.0
		FNeg = 0.0
		Tsdata = DataSet[-SetNum:]
		Trdata = DataSet[0:-SetNum]
		meanvalue, result, a_0, a_1 = his(Trdata)
		ACC, TP, FP, TN, FN = test2(TestData,meanvalue, result, a_0, a_1, x)
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
	plt.title('Curve')
	plt.show()



DataSet = loadData('/Users/skylatitude/Desktop/spambase/spambase.data')
DataSet = ShuffleData(DataSet)
SetNum = (len(DataSet) -1 ) / 10
for i in range(10):
	TestData = DataSet[SetNum*i:(SetNum*i + SetNum)]
	TrainData = DataSet[0: SetNum*i]
	TrainData.extend(DataSet[(SetNum*i + SetNum):])
	meanvalue, result, a_0, a_1 = his(TrainData)
	acc = test(TestData,meanvalue, result, a_0, a_1)
	print 'The ACC of %d Folder is %f' %(i+1,acc)
roc(DataSet, SetNum)




