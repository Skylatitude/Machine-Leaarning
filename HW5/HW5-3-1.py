from numpy import *
import random 
from math import log


def loadData(filename):
	vector = []
	with open(filename) as file:
		for line in file:
			vector.append(map(float,line.split()))
	return vector

def Normalize(TrainData, TestData):
	data = TrainData[:]
	data.extend(TestData)
	for i in range(len(TrainData[0])):
		featurelist = [vector[i] for vector in data]
		minnum = min(featurelist)
		maxnum = max(featurelist)
		for j in range(len(featurelist)):
			if maxnum - minnum == 0:
				featurelist[j] = (featurelist[j] - minnum)/1e-20
			else:
				featurelist[j] = (featurelist[j] - minnum)/(maxnum - minnum)
		for j in range(len(data)):
			data[j][i] = featurelist[j]
	return data[:len(TrainData)], data[len(TrainData):]

def GDlogistc(TrainData, TrainLabel, Lambda, k):
	m, n = shape(TrainData)
	w = mat(ones((n,1)))
	last = mat(ones((n,1)))
	# k = 0.0
	while (1):
		last = mat(TrainData) * w
		for j in range(n):
			for i in range(m):
				temp = 1.0/(1 + exp(- TrainData[i] * w))
				w[j, 0] += Lambda * (TrainLabel[i] - temp) * TrainData[i, j] - k/m * w[j,0]

				#err = res(TrainData, TrainLabel, w)
		# 		print i
		# 		if sum(multiply((mat(TrainLabel) - mat(TrainData)*w),(mat(TrainLabel) - mat(TrainData)*w)),)/len(last) <= 1e-20:		
		# 			break
		# 	break
		# break
		if err < 0.15:
			break
		# if k != 0:
		# 	if err < 0.15:
		# 		break
		# else:
		# 	if err < 0.12:
		# 		break
		# print err
	return w, k

def sigmoid(z):
	return 1.0 / (1.0 + exp(-z))

def res(DataSet, LabelSet, w):
	x = mat(DataSet)
	y = mat(LabelSet)
	predict = x * w
	err = 0.0
	for i in range(len(DataSet)):
		predict[i] = sigmoid(predict[i])
		if (predict[i]>=0.4 and y[i] == 0) or (predict[i]<0.4 and y[i] == 1):
			err += 1
	return err/len(DataSet)

TrainData = loadData('/Users/skylatitude/downloads/spam_polluted/train_feature.txt')
TrainLabel = loadData('/Users/skylatitude/downloads/spam_polluted/train_label.txt')
TestData = loadData('/Users/skylatitude/downloads/spam_polluted/test_feature.txt')
TestLabel = loadData('/Users/skylatitude/downloads/spam_polluted/test_label.txt')
Train, Test  =  Normalize(TrainData, TestData)
# print Train
# print Test
print shape(Train), shape(Test)
w, k  = GDlogistc(mat(Train), mat(TrainLabel), 5, 12)
err = res(mat(Test), mat(TestLabel), w)
print 'The Test ACC is ', 1 - err


