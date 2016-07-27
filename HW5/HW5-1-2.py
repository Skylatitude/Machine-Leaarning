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
	return mat(traindata), trainlabel

def buildTree(DataSet, weight):
	traindata, trainlabel = SplitDataSet(DataSet)
	datamatrix = mat(traindata)
	labelmatrix = mat(trainlabel).T
	m, n = shape(datamatrix)
	bestStump = {}
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

def buildTree2(DataSet, label, weight, record):
    m, n = shape(DataSet)
    datamatrix = mat(DataSet)
    # labelmatrix = mat(label)
    bestStump = {}
    matlabel = mat(label)
    bestclass = zeros((m, 1))
    minErr = inf
    p = random.randint(0, m - 1)
    q = random.randint(0, n - 1)
    #print p, q
    while q in record:
        q = random.randint(0, n - 1)
    bestStump['datapoint'] = p
    bestStump['feature'] = q
    bestStump['thre'] = datamatrix[p, q]
    bestStump['label'] = label[p]
    for mark in ['small', 'large']:
        predict = classify(datamatrix, q, datamatrix[p, q], mark)
        # print shape(predict)
        # print shape([label])
        # print predict
        errArr = mat(ones((m, 1)))
        errArr[predict == matlabel] = 0
        weighterr = weight.T * errArr
        if weighterr < minErr:
            minErr = weighterr
            bestclass = predict.copy()
            bestStump['chose'] = mark
    return bestStump, minErr, bestclass, p, q




def adaboost(DataSet, Label):
    result = []
    datanum = len(DataSet)
    weight = mat(ones((datanum, 1))) / datanum
    predict = mat(zeros((datanum, 1)))
    iteration = 0
    temp = mat(ones((datanum, 1)))
    a = 0
    record = []
    while (1):
        errArr = 0.0
        bestStump, error, bestclass, p, q = buildTree2(DataSet, Label, weight, record)
        # print sum(bestclass), sum(Label)
        alpha = 0.5 * log((1.0 - error) / max(error, 1e-10))
        bestStump['alpha'] = alpha
        #print 'alpha', alpha
        result.append(bestStump)
        #print sum(weight)
        # for i in range(datanum):
        #     if bestclass[i] == Label[i]:
        #         weight[i] = weight[i] * exp(-alpha)
        #     else:
        #         weight[i] = weight[i] * exp(alpha)
        expon = multiply(-1 * alpha * mat(Label), bestclass)
        weight = multiply(weight, exp(expon))


        weight = weight / sum(weight)

        #print sum(bestclass)
        #for i in range(datanum):
        predict += alpha * bestclass
        #print predict
        # for i in range(datanum):
        #     # print "sign:",sign(predict[i]), "Label:",Label[i], "equal: ", sign(predict[i]) == Label[i]
        #     if sign(predict[i]) != Label[i]:
        #         errArr += 1
        # print errArr
        errArr = sum((multiply(mat(Label), (sign(predict))) - mat(ones((datanum, 1)))) / -2)
        errate = errArr / datanum
        iteration += 1
        #print iteration
        b, predictval = adaboosttest(DataSet, Label, result)
        #if errate - a <= 0.01:
        #    record.append(q)
        #a = errate
        #print 'The acc is %f' % (1 - errate)
        if (1 - errate) >= 0.996 or iteration >= 3000:
            break
    return result
	
def adaboosttest(testdata, testlabel, result):
    m, n = shape(testdata)
    acc = 0.0
    # testtrun = []
    testresult = zeros((m, 1))
    for i in range(len(result)):
        pre = classify(testdata, result[i]['feature'], result[i]['thre'], result[i]['chose'])
        testresult += mat(pre) * result[i]['alpha']
    for i in range(m):
        if sign(testresult[i]) == testlabel[i]:
            acc += 1.0
    print 'The acc is %f' % (acc / m)
    # print testresult
    return sign(testresult), (acc / m)


TrainData = loadData('/Users/skylatitude/downloads/spam_polluted/train_feature.txt')
TrainLabel = loadData('/Users/skylatitude/downloads/spam_polluted/train_label.txt')
TestData = loadData('/Users/skylatitude/downloads/spam_polluted/test_feature.txt')
TestLabel = loadData('/Users/skylatitude/downloads/spam_polluted/test_label.txt')
# for i in range(len(TrainData)):
# 	TrainData[i].extend(TrainLabel[i])
# for i in range(len(TestData)):
# 	TestData[i].extend(TestLabel[i])
# DataSet = ShuffleDataSet(DataSet)
for i in TrainLabel:
	if i[0] == 0:
		i[0] = -1
# testlabel = []
# for i in TestLabel:
# 	if i == 0:
# 		testlabel.append(-1)
# 	else:
# 		testlabel.append(i)
topten = []
result = adaboost(mat(TrainData), TrainLabel)
# final = sorted(margin.items(), key=lambda x:x[1], reverse = True)[:15]
# for i in range(len(final)):
# 	topten.append(final[i][0])
# print topten[:10]









