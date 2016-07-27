from numpy import *
import random 
from math import log


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
					bestStump['list'] = shortlist
	return bestStump, minErr, bestClass, labelmatrix

def RandTree(DataSet, weight):
	traindata, trainlabel = SplitDataSet(DataSet)
	datamatrix = mat(traindata)
	labelmatrix = mat(trainlabel).T
	m, n = shape(datamatrix)
	bestStump = {};
	bestClass = mat(zeros((m,1)))
	minErr = inf
	p = random.randint(0, m - 1)
	q = random.randint(0, n - 1)
	for where in ['small', 'large']:
		threshold = traindata[p][q]
		predict = classify(datamatrix, q, threshold, where)
		errArr = mat(ones((m,1)))
		errArr[predict == labelmatrix] = 0
		weighterr = weight.T * errArr
		if weighterr < minErr:
			minErr = weighterr
			bestClass = predict.copy()
			bestStump['feature'] = q
			bestStump['thres'] = threshold
			bestStump['chose'] = where
			# bestStump['list'] = shortlist
	return bestStump, minErr, bestClass, labelmatrix

def adaboost(DataSet, TestData):
	result = []
	datanum = len(DataSet)
	weight = mat(ones((datanum, 1))) / datanum
	predict = mat(zeros((datanum, 1)))
	iteration = 0
	margin = {}
	yst = [0.0]*datanum
	while (1):
		errArr = []
		epsilon = 0.0
		bestStump, error, bestClass, Label = RandTree(DataSet, weight)
		alpha = 0.5 * log((1.0 - error)/ max(error, 1e-10))
		bestStump['alpha'] = alpha
		for i in range(len(DataSet)):
			yst[bestStump['feature']] += alpha * Label[i] * bestClass[i]
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
		print iteration, errate, error[0, 0]
		predictval = adaboosttest(TestData, result)
		if errate == 0.0 or iteration >= 300: break
		for i in range(datanum):
			margin[i] = yst[i]
	return result, margin
	
def adaboosttest(dataTest, result):
	testdata, testlabel = SplitDataSet(dataTest)
	testmatrix = mat(testdata)
	m, n = shape(testdata)
	err = 0.0
	testresult = mat(zeros((m, 1)))
	listresult = []
	for i in range(len(result)):
		pre = classify(testmatrix, result[i]['feature'], result[i]['thres'], result[i]['chose'])
		testresult += mat(pre) * result[i]['alpha']
	for i in range(m):
		if sign(testresult[i]) == testlabel[i]:
			listresult.append(testresult[i,0])
			err += 1.0
	print 'The acc is %f' %(err/m)
	return sign(testresult)

DataSet = loadData('/Users/skylatitude/Desktop/spambase/spambase.data')
DataSet = ShuffleDataSet(DataSet)
topten = []
SetNum = (len(DataSet) - 1)/10
TestData = DataSet[0:SetNum]
TrainData = DataSet[SetNum:]
result, margin = adaboost(TrainData, TestData)
final = sorted(margin.items(), key=lambda x:x[1], reverse = True)[:15]
for i in range(len(final)):
	topten.append(final[i][0])
print topten[:10]









