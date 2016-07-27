from numpy import *
import random
# import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def loadData(filename):
    vector = []

    with open(filename) as file:
        for line in file:
            vector.append(map(float, line.split()))
    # fe = [[]*len(vector[0])]*len(vector)
    for i in range(len(vector)):
    	for j in range(len(vector[0])):
    		if vector[i][j] < inf:
    			vector[i][j] = vector[i][j]
    		else:
    			vector[i][j] = 0.0
    return vector

def loadData222(filename):
    # vector = []

    dataset = []
    try:
        f = open(filename, 'r')
    except IOError:
        print "This file does not exist"
    for line in f:
        dataset.append(map(float, line.split(',')))
    return dataset

def ShuffleData(DataSet):
	
	random.shuffle(DataSet)
	return DataSet

def Prob(DataSet):
	# pro = []
	# a0 = 0.0
	# a1 = 0.0
	result = []
	meanvalue = []
	
	DataNum = len(DataSet)
	featureNum = len(DataSet[0]) - 1
	for i in range(featureNum):	
		Featurelist = []
		a_0 = 0.0
		a_1 = 0.0
		p1 = 0.0 #<=mean value/spam
		p2 = 0.0 # >mean value/spam
		p3 = 0.0 #<=mean value/non-spam
		p4 = 0.0 # >mean value/non-spam
		for data in DataSet:
			if data[i] < inf:
				Featurelist.append(data[i])
			else:
				Featurelist.append(0)
			# if data[-1] == 1:
			# 	a1 += 1
			# else:
			# 	a0 += 1
		mv = mean(Featurelist)
		# print Featurelist
		# print mv
		meanvalue.append(mv)
		for data in DataSet:
			# print 'label: ', data[-1]
			if data[-1] == 1.0 and data[i] < mv:
				p1 += 1
				a_1 += 1
			if data[-1] == 1.0 and data[i] >= mv:
				p2 += 1
				a_1 += 1
			if data[-1] == 0.0 and data[i] < mv:
				p3 += 1
				a_0 += 1
			if data[-1] == 0.0 and data[i] >= mv:
				p4 +=1
				a_0 += 1
		# print a_1, a_0, p1, p2, p3, p4
		p1 = p1 / a_1
		p2 = p2 / a_1
		p3 = p3 / a_0
		p4 = p4 / a_0
		result.append([p1,p2,p3,p4])
		# pro.append([a_0/(a_0+a_1), a_1/(a_0+a_1)])
	return meanvalue, result, a_0/len(DataSet), a_1/len(DataSet)

def test(DataSet, meanvalue, result, a_0, a_1):
	predict = []
	acc =0.0
	s = 0
	for x in range(len(DataSet)):
		p_1 = 1
		p_0 = 1
		for i in range(len(DataSet[0])-1):
			if DataSet[x][i] <= meanvalue[i]:
				p_1 = p_1 * result[i][0]
				p_0 = p_0 * result[i][2]
			else:
				p_1 = result[i][1] * p_1
				p_0 = result[i][3] * p_0
			# else:
			# 	s += 1
		if p_1 * a_1 > p_0 * a_0:
			predict.append(1.0)

		else :
			predict.append(0.0)
		if DataSet[x][-1] == predict[-1]:
			acc += 1
			# print x
	# print predict
	return acc / len(DataSet)
# def test2(DataSet, meanvalue, result, a_0, a_1):

# 	lnp = []
# 	for data in DataSet:
# 		p_1 = 1
# 		p_0 = 1
# 		for i in range(len(DataSet[0])-1):
# 			if data[i] <= meanvalue[i]:
# 				p_1 = p_1 * result[i][0]
# 				p_0 = p_0 * result[i][2]
# 			else:
# 				p_1 = result[i][1] * p_1 
# 				p_0 = result[i][3] * p_0 
# 		if p_1 == 0 or p_0 ==0:
# 			p_1 = p_1+10**(-100)
# 			p_0 = p_0+10**(-100)
# 		p = math.log((p_1*a_1/p_0*a_0),10)
# 		lnp.append(p)
# 	return lnp
# def pic(DataSet, lnp, x):
# 	predict = []
# 	acc =0.0
# 	TP = 0
# 	TN = 0
# 	FP = 0
# 	FN = 0
# 	for i in range(len(DataSet)):
# 		if lnp[i] > min(lnp) + x*((max(lnp) - min(lnp) + 10)/200):
# 			predict.append(1)

# 		else :
# 			predict.append(0)
# 		if predict[i] == 1 and DataSet[i][-1] == 1:
# 			TP += 1
# 			acc += 1
# 		if predict[i] == 1 and DataSet[i][-1] == 0:
# 			FP += 1
# 		if predict[i] == 0 and DataSet[i][-1] == 0:
# 			TN += 1
# 			acc += 1
# 		if predict[i] == 0 and DataSet[i][-1] == 1:
# 			FN += 1
# 	print 'TP = %d FP = %d TN = %d FN = %d' %(TP, FP, TN, FN)
# 	return acc / len(DataSet), TP, FP, TN, FN

# def roc(DataSet, SetNum):
# 	FPR = []
# 	TPR = []
# 	Tsdata = DataSet[-SetNum:]
# 	Trdata = DataSet[0:-SetNum]
# 	meanvalue, result, a_0, a_1 = Prob(Trdata)
# 	lnp = test2(Tsdata, meanvalue, result, a_0, a_1)
# 	for x in range(200):
# 		TPos = 0.0
# 		FPos = 0.0
# 		TNeg = 0.0
# 		FNeg = 0.0
# 		ACC, TP, FP, TN, FN = pic(Tsdata,lnp,x)
# 		TPos += TP
# 		FPos += FP 
# 		TNeg += TN 
# 		FNeg += FN
# 		TPR.append(TPos/(TPos + FNeg))
# 		FPR.append(FPos/(FPos + TNeg))
# 	for i in range(200):
# 		plt.plot(FPR[i], TPR[i], 'b*')
# 		plt.plot(FPR, TPR)
# 	a = sum(FPR)
# 	b = sum(TPR)
# 	AUC = -trapz(TPR,FPR)
# 	print AUC
# 	plt.title('Bernoulli Curve')
# 	plt.show()

TrainData = loadData('/Users/skylatitude/downloads/20_percent_missing_train.txt')
# TrainLabel = loadData('/Users/skylatitude/downloads/spam_polluted/train_label.txt')
TestData = loadData222('/Users/skylatitude/downloads/20_percent_missing_test.txt')
# TestLabel = loadData('/Users/skylatitude/downloads/spam_polluted/test_label.txt')

# print TestData

# trainnum = len(TrainData)
# DataSet = TrainData[:]
# print shape(TrainData)
# print '############################################################'
# print shape(TestData)
# DataSet.extend(TestData)
# DataSetlowD = PCA(n_components=100)
# Data = DataSetlowD.fit(DataSet).transform(DataSet)
# print shape(Data)
# print Data
# Train = Data[0:trainnum]
# Test = Data[trainnum:]
# for i in range(len(Train)):
# 	TrainData[i].extend(TrainLabel[i])
# for i in range(len(Test)):
# 	TestData[i].extend(TestLabel[i])
meanvalue, result, a_0, a_1 = Prob(TrainData)
print meanvalue
acc = test(TestData, meanvalue, result, a_0, a_1)
print 'The ACC of Test Folder is %f' %(acc)

