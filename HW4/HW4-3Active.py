from numpy import *
import random 
from math import log
import matplotlib.pyplot as plt

def loadData(filename):
	vector = []
	with open(filename) as file:
		for line in file:
			vector.append(map(float,line.split()))
	return vector

def ShuffleDataSet(DataSet):
	random.shuffle(DataSet)
	return DataSet

def classify(DataSet, colum, threshold, mark):
	predictArray = ones((shape(DataSet)[0],1))
	if mark == 'small':
		predictArray[DataSet[:,colum] <= threshold] = -1
	else:
		predictArray[DataSet[:,colum] > threshold]  = -1
	return predictArray

def SplitDataSet(DataSet):
	traindata = []
	trainlabel = []
	for data in DataSet:
		if data[-1] == 0:
			trainlabel.append(-1)
		else:
			trainlabel.append(data[-1])
		traindata.append(data[:-1])
	return traindata, trainlabel

def buildTree(DataSet, weight):
	traindata, trainlabel = SplitDataSet(DataSet)

	datamatrix = mat(traindata)
	labelmatrix = mat(trainlabel).T
	m, n = shape(datamatrix)
	bestStump = {};
	bestClass = mat(zeros((m,1)))
	minErr = inf
	for i in range(n):
		featurelist = []
		for data in traindata:
			featurelist.append(data[i])
		t = set(featurelist)
		shortlist = list(t)
		shortlist.append((max(shortlist)+1))
		shortlist.append((min(shortlist)-1))
		for j in range(len(shortlist)):
			for where in ['small', 'large']:
				threshold = shortlist[j]
				predict = classify(datamatrix, i, threshold, where)
				errArr = mat(ones((m,1)))
				errArr[predict == labelmatrix] = 0
				weighterr = weight.T * errArr
				if weighterr < minErr:
					minErr = weighterr
					bestClass = predict.copy()
					bestStump['feature'] = i
					bestStump['thres'] = threshold
					bestStump['chose'] = where
	return bestStump, minErr, bestClass, labelmatrix

# def adaboost(DataSet):
# 	result = []
# 	datanum = len(DataSet)
# 	predict = mat(zeros((datanum,1)))
# 	weight = mat(ones((len(DataSet),1))/datanum)
# 	iteration = 0
# 	while(1):
# 		a = 0
# 		b = 0
# 		errArr = []
# 		epsilon = 0.0
# 		bestStump, error, bestClass, Label = buildTree(DataSet, weight)
# 		for i in range(len(bestClass)):
# 			if bestClass[i] != Label[i]:
# 				epsilon += weight[i]
# 		print weight
# 		alpha = 0.5 * log((1.0 - epsilon)/ max(epsilon, 1e-10))
# 		# print alpha
# 		bestStump['alpha'] = alpha
# 		result.append(bestStump)
# 		for i in range(len(bestClass)):
# 			if bestClass[i] != Label[i]:
# 				x = exp(-1.0 * alpha)
# 			else:
# 				a += 1
# 				x = exp(alpha)
# 		# print weight
# 			weight[i] = weight[i] * x
# 		weight = weight / sum(weight)
# 		predict += alpha * bestClass
# 		print shape(predict)
# 		for i in range(len(Label)):
# 			if predict[i] <= 0.5 and Label[i] == 0:
# 				predict[i] = 0.0
# 				errArr.append(0.0)
# 				b += 1
# 			if predict[i] <= 0.5 and Label[i] != 0:
# 				predict[i] = 0.0
# 				errArr.append(1.0)
# 			if predict[i] > 0.5 and Label[i] == 1:
# 				predict[i] = 1
# 				errArr.append(0.0)
# 				b += 1
# 			if predict[i] > 0.5 and Label[i] != 1:
# 				predict[i] = 1
# 				errArr.append(1.0)
# 		errate = sum(errArr) / datanum
# 		iteration += 1
# 		print iteration, errate
# 		print a, b
# 		if errate == 0.0:
# 			break
# 	return result

def adaboost(DataSet):
	result = []
	datanum = len(DataSet)
	weight = mat(ones((datanum, 1))) / datanum
	predict = mat(zeros((datanum, 1)))
	iteration = 0
	temp = mat(ones((datanum, 1)))
	while (1):
		errArr = []
		epsilon = 0.0
		bestStump, error, bestClass, Label = buildTree(DataSet, weight)
		# for i in range(len(bestClass)):
		# 	if bestClass[i] != Label[i]:
		# 		epsilon += weight[i]
		# print epsilon == error
		alpha = 0.5 * log((1.0 - error)/ max(error, 1e-10))
		bestStump['alpha'] = alpha
		result.append(bestStump)
		for i in range(datanum):
			if bestClass[i] == Label[i]:
				weight[i] = weight[i] * exp(-alpha)
			else:
				weight[i] = weight[i] * exp(alpha)
		weight = weight / sum(weight)
		for i in range(datanum):
			predict[i] += alpha * bestClass[i]
		for i in range(datanum):
			if sign(predict[i]) == Label[i]:
				errArr.append(0.0)
			else:
				errArr.append(1.0)
		errate = sum(errArr) / datanum
		iteration += 1
		# print iteration, errate, error
		if errate == 0.0 or iteration >= 10: break
	# print 'The Train Error is %f' %(errate)
	return result
	
def adaboosttest(dataTest, result, step):
	testdata, testlabel = SplitDataSet(dataTest)
	testmatrix = mat(testdata)
	m, n = shape(testdata)
	err = 0.0
	testresult = mat(zeros((m, 1)))
	ch = {}
	addtrain = []
	for i in range(len(result)):
		pre = classify(testmatrix, result[i]['feature'], result[i]['thres'], result[i]['chose'])
		testresult += mat(pre) * result[i]['alpha']
	for i in range(m):
		if sign(testresult[i]) == testlabel[i]:
			err += 1.0
		ch[abs(testresult[i])] = i
	f = sorted(ch.items())
	for i in range(step):
		addtrain.append(f[i][1])
	# print 'The acc is %f' %(err/m)
	return sign(testresult), (m - err)/m, addtrain


DataSet = loadData('/Users/skylatitude/Desktop/spambase/spambase.data')
DataSet = ShuffleDataSet(DataSet)
SetNum = (len(DataSet) - 1)/10
step = 2 * SetNum / 10
TrainData = DataSet[0:int(SetNum/2)]
TestData = DataSet[int(SetNum/2):]
# TestData = DataSet[0:SetNum]
for i in range(23):
	result = adaboost(TrainData)
	testresult, err, addlist = adaboosttest(TestData, result, step)
	addlist = sorted(addlist)
	addlist = list(reversed(addlist))
	for m in range(len(addlist)):
		index = addlist[m]
		t = TestData[index]
		TrainData.append(t)
		del TestData[index]
	print len(TestData)
	# temp = []
	# for m in range(len(TestData)):
	# 	for j in range(len(addlist)):
	# 		if m != addlist[j]:
	# 			temp.append(TestData[m])
	# TestData = temp
	print shape(TrainData),shape(TestData)
	print 'The test err is : %f' %(err)
	if i == 0 :
		print 'The 5%% Training Set\'s Test Error is: %f' %(err)
	if i == 10 :
		print 'The 25%% Training Set\'s Test Error is: %f' %(err)
	if i == 22 :
		print 'The 50%% Training Set\'s Test Error is: %f' %(err)

