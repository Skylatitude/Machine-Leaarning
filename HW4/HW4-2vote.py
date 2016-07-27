from numpy import *
import random 
from math import log
import matplotlib.pyplot as plt

def loadData(filename):
	vector = []
	with open(filename) as file:
		for line in file.readlines():
			# t = line.translate(None, '\t\n')
			t = line.split()
			vector.append(t)
	return vector

def ShuffleDataSet(DataSet):
	random.shuffle(DataSet)
	return DataSet

def classify(DataSet, colum, threshold, mark):
	predictArray = ones((shape(DataSet)[0],1))
	if mark == 'small':
	# 	for i in range(len(DataSet)):
	# 		if DataSet[i,colum] == '?':
	# 			if i % 2 == 0:
	# 				predictArray[i] = -1
	# 		if DataSet[i,colum] == threshold:
	# 			predictArray[i] = -1
	# else:
	# 	for i in range(len(DataSet)):
	# 		if DataSet[i,colum] == '?':
	# 			if i % 2 == 0:
	# 				predictArray[i] = -1
	# 		if DataSet[i,colum] != threshold:
	# 			predictArray[i] = -1


		predictArray[DataSet[:,colum] <= threshold] = -1
	else:
		predictArray[DataSet[:,colum] > threshold]  = -1
	return predictArray

# def plot(DataSet, colum, mark):
# 	predictArray = ones((shape(DataSet)[0],1))
# 	if mark == 'small':
# 		predictArray[DataSet[:,colum] <= threshold] = -1
# 	else:
# 		predictArray[DataSet[:,colum] > threshold]  = -1
# 	return predictArray

def SplitDataSet(DataSet):
	traindata = []
	trainlabel = []
	for data in DataSet:
		if data[-1] == 'd':
			trainlabel.append(-1)
		else:
			trainlabel.append(1)
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
		# featurelist = []
		# for data in traindata:
		# 	featurelist.append(data[i])
		# t = set(featurelist)
		# shortlist = list(t)
		# shortlist.append((max(shortlist)+1))
		# shortlist.append((min(shortlist)-1))
		for threshold in ['y', 'n']:
			for where in ['small', 'large']:
				# threshold = shortlist[j]
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
		
		if errate == 0.0 or iteration >= 15: break

	return result
	
def adaboosttest(dataTest, result):
	testdata, testlabel = SplitDataSet(dataTest)
	testmatrix = mat(testdata)
	m, n = shape(testdata)
	err = 0.0
	testresult = mat(zeros((m, 1)))
	for i in range(len(result)):
		pre = classify(testmatrix, result[i]['feature'], result[i]['thres'], result[i]['chose'])
		testresult += mat(pre) * result[i]['alpha']
	for i in range(m):
		if sign(testresult[i]) == testlabel[i]:
			err += 1.0
	# print 'The test acc is %f' %(err/m)
	return sign(testresult), err/m


DataSet = loadData('/Users/skylatitude/downloads/vote/vote.data')
DataSet = ShuffleDataSet(DataSet)
SetNum = (len(DataSet) - 1)/20
for i in range(1,10):
	TrainData = DataSet[0:SetNum*i]
	TestData = DataSet[SetNum*i:]
	result = adaboost(TrainData)
	predictval, err = adaboosttest(TestData, result)
	print '%d of training data set acc is: %f' %(i*5, err)
