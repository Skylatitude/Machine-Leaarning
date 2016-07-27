import csv
from numpy import *
import random



# def loaddata(filename):
# 	header = [n for n in range(1754)]
# 	with open(filename) as f:
# 	    f_csv = csv.DictReader(f)
# 	    for row in f_csv:
# 	    	print row
# 	    	break
	    # f_csv.writeheader()
	    # f_csv.writerows(filename)

def classify(DataSet, colum, threshold, mark):
	# print DataSet[0, colum]
	print threshold
	predictArray = ones((len(DataSet),1))
	print 'predictArray before:',sum(predictArray)
	if mark == 'small':
		# for i in range(len(DataSet)):
		# 	if DataSet[i,colum] <= threshold:
		# 		predictArray[i] = -1
		predictArray[DataSet[:,colum] <= threshold] = -1
	else:
		predictArray[DataSet[:,colum] > threshold] = -1
	# 	if int()
	# 		predictArray[i] = -1
	print 'predictArray after:',sum(predictArray)
	return predictArray

def creatFunctionTable():
	table = mat(ones((8, 20)))
	tableinfo = []
	for i in range(20):
		f = []
		numofFeature = random.randint(1,7)
		for j in range(numofFeature):
			t = random.randint(0,7)
			table[t, i] = -1
			f.append(t)
		
		for n in range(i):
			if f == tableinfo[n]:
				m = random.randint(0,7)
				table[m, i] = -1
				f.append(m)
				continue
		tableinfo.append(list(set(f)))
	return table, tableinfo

def loadData(filename):
	vector = zeros((11314, 1754))
	# tablelabel = mat(ones((11314, 20)))
	label = []
	n = 0
	with open(filename) as file:
		for line in file.readlines():
			# t = line.translate(None, '\t\n')
			t = line.split()
			label.append(t[0])
			# for i in range(len(tableinfo)):
			# 	if t[0] not in tableinfo:
			# 		tablelabel[n, i] = -1
			for i in range(1,len(t)):		
				temp = t[i].split(':')
				vector[n, temp[0]] = float(temp[1])
			n += 1
	return vector, label

def loadTestData(filename):
	vector = mat(zeros((7532, 1754)))
	testtable = mat(ones((7532, 20)))
	label = []
	n = 0
	with open(filename) as file:
		for line in file.readlines():
			# t = line.translate(None, '\t\n')
			t = line.split()
			label.append(t[0])
			# for i in range(len(tableinfo)):
			# 	if t[0] not in tableinfo:
			# 		tablelabel[n, i] = -1
			for i in range(1,len(t)):		
				temp = t[i].split(':')
				vector[n, temp[0]] = temp[1]
			n += 1
	return vector, label

def buildTree(DataSet, label, weight):
	m, n = shape(DataSet)
	datamatrix = mat(DataSet)
	# labelmatrix = mat(label)
	bestStump = {}
	matlabel = mat([label])
	bestclass = zeros((m,1))
	minErr = inf
	p = random.randint(0, m - 1)
	q = random.randint(0, n - 1)
	print p, q
	bestStump['datapoint'] = p
	bestStump['feature'] = q
	bestStump['thre'] = DataSet[p, q]
	bestStump['label'] = label[p]
	for mark in ['small', 'large']:
		predict = classify(datamatrix, q, DataSet[p,q], mark)
		# print shape(predict)
		# print shape([label])
		# print predict
		errArr = mat(ones((m,1)))
		errArr[predict == matlabel.T] = 0
		weighterr = weight.T * errArr
		if weighterr < minErr:
			minErr = weighterr
			bestclass = predict.copy()
			bestStump['chose'] = mark
	return bestStump, minErr, bestclass

# def buildTree(DataSet, label, weight):
# 	datamatrix = mat(DataSet)
# 	labelmatrix = mat(label).T
# 	m, n = shape(DataSet)
# 	bestStump = {};
# 	bestClass = mat(zeros((m,1)))
# 	minErr = inf
# 	err = 0.0
# 	for i in range(n):
# 		featurelist = []
# 		for j in range(m):
# 			featurelist.append(int(DataSet[j,i]))
# 		t = set(featurelist)
# 		shortlist = list(t)
# 		print shape(shortlist)
# 		# shortlist.append((max(shortlist)+1))
# 		# shortlist.append((min(shortlist)-1))
# 		for j in range(len(shortlist)):
# 			for mark in ['small', 'large']:
# 				threshold = shortlist[j]
# 				print 'fffffffffffffffffffffffffffff'
# 				print threshold
# 				predict = classify(DataSet, i, threshold, mark)
# 				# print shape(predict)
# 				# print shape([label])
# 				print sum(predict)
# 				errArr = mat(ones((m,1)))
# 				x = mat(label)
# 				for p in range(len(label)):
# 					if predict[p] == label[p]:
# 						errArr[p] = 0
# 				weighterr = weight.T * errArr
# 				if weighterr < minErr:
# 					minErr = weighterr
# 					bestclass = predict.copy()
# 					bestStump['chose'] = mark
# 					bestStump['feature'] = i
# 					bestStump['thres'] = threshold
# 					bestStump['chose'] = where
# 	return bestStump, minErr, bestClass

def adaboost(DataSet, Label):
	result = []
	datanum = len(DataSet)
	weight = mat(ones((datanum, 1))) / datanum
	predict = mat(zeros((datanum, 1)))
	iteration = 0
	temp = mat(ones((datanum, 1)))
	while (1):
		errArr = []
		bestStump, error, bestclass = buildTree(DataSet, Label, weight)
		# print sum(bestclass), sum(Label)
		alpha = 0.5 * log((1.0 - error) / max(error, 1e-10))
		bestStump['alpha'] = alpha
		print 'alpha', alpha
		result.append(bestStump)
		print sum(weight)
		for i in range(datanum):
			if bestclass[i] == Label[i]:
				weight[i] = weight[i] * exp(-alpha)
			else:
				weight[i] = weight[i] * exp(alpha)

		weight = weight / sum(weight)

		print sum(bestclass)
		for i in range(datanum):
			predict[i] += alpha * bestclass[i]
		print predict
		for i in range(datanum):
			# print "sign:",sign(predict[i]), "Label:",Label[i], "equal: ", sign(predict[i]) == Label[i]
			if sign(predict[i]) == Label[i]:
				errArr.append(0.0)
			else:
				errArr.append(1.0)
		errate = sum(errArr) / datanum
		iteration += 1
		print iteration, sum(errArr), error
		# predictval = adaboosttest(TestData, Label, result)
		if errate <= 0.1 : break
	return result

def adaboosttest(testdata, testlabel, result):
	m, n = shape(testdata)
	err = 0.0
	# testtrun = []
	testresult = zeros((m, 1))
	for i in range(len(result)):
		pre = classify(testdata, result[i]['feature'], result[i]['thre'], result[i]['chose'])
		testresult += mat(pre) * result[i]['alpha']
	for i in range(m):
		if sign(testresult[i]) == testlabel[i]:
			err += 1.0
	print 'The acc is %f' %(err/m)
	# print testresult
	return sign(testresult)

def GetFunctionLabel(tableinfo, Label):
	labelArr = zeros((len(tableinfo), len(Label)))
	
	# for j in range(len(table[0])):
	# 	funclabel = mat(zeros((len(Label), 1)))
	# 	for i in range(len(table)):
	# 		if table[i, j] == 1:
				
	# 			templabel = mat(ones((len(Label), 1))) * j
	# 			funclabel[templabel == Label] = 1
	# 	labelArr[:,j] = funclabel.T
	# print tableinfo
	for i in range(len(tableinfo)):
		arrayt = []
		flag = 0
		for j in range(len(Label)):
			# for n in range(len(tableinfo[i])):
			if int(Label[j]) in tableinfo[i]:
				labelArr[i][j] = 1
			else:
				labelArr[i][j] = -1
			# print Label[j]
			if Label[j] == '0':
				flag += 1
	return labelArr



table, tableinfo = creatFunctionTable()
traindata, trainlabel = loadData('/Users/skylatitude/downloads/8newsgroup/train.trec/feature_matrix.txt')
testdata, testlabel = loadTestData('/Users/skylatitude/downloads/8newsgroup/test.trec/feature_matrix.txt')
# print shape(trainlabel)
print max(traindata[0,:])
print shape(traindata)
# print int(trainlabel[1112]) == tableinfo[0][0]


# print len(tableinfo)
# translabel = GetFunctionLabel(table, trainlabel)
# print set(trainlabel), set(trainlabel)
# print tableinfo, shape(tableinfo), type(tableinfo)
trainlabelArr = GetFunctionLabel(tableinfo, trainlabel)
testlabelArr  = GetFunctionLabel(tableinfo, testlabel)
test = []
# print trainlabelArr
# print testlabelArr
for i in range(20):
	result = adaboost(traindata, trainlabelArr[i])
	testpredict = adaboosttest(testdata, testlabelArr[i], result)
	print type(testpredict), shape(testpredict)
	# print testpredict
	test.append(testpredict)
# print shape(test)
# xxx = mat(test)
acc = 0.0
for i in range(len(testlabel)):
	f = []
	for n in range(20):
		f.append(test[n][i][0])
	# print f
		# print table[j] - mat(f)
	print sum(table[int(testlabel[i])] - mat(f))
	if sum(table[int(testlabel[i])] - mat(f))<=7:
		acc += 1
print acc / len(testlabel)










