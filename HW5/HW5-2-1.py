from numpy import *
import random
import scipy.stats as stats
from sklearn.decomposition import PCA

def loadData(filename):
	vector = []
	with open(filename) as file:
		for line in file:
			vector.append(map(float,line.split()))
	return vector

def ShuffleData(DataSet):
	
	random.shuffle(DataSet)
	return DataSet

def Normal(DataSet,LabelSet):

	meanvalue = []
	variance  = []
	vari1 = 0.0
	vari0 = 0.0
	a_1 = 0.0
	a_0 = 0.0
	for data in LabelSet:
		if data[0] == 1:
			a_1 += 1
		else:
			a_0 += 1
	a_1 = a_1 / len(DataSet)
	a_0 = a_0 / len(DataSet)
	# print a_0, a_1
	DataNum = len(DataSet)
	featureNum = len(DataSet[0])
	for i in range(featureNum):
		FeatureList1 = []
		FeatureList0 = []
		FeatureList  = []
		for j in range(len(DataSet)):
			FeatureList.append(DataSet[j][i])
			if LabelSet[j][0] == 1:
				FeatureList1.append(DataSet[j][i])
			else:
				FeatureList0.append(DataSet[j][i])
		# print len(FeatureList0), len(FeatureList1)
		mv0 = mean(FeatureList0)
		mv1 = mean(FeatureList1)
		# print 'mean  : ', mv0, mv1
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
		# print 'variance : ', vari0, vari1, vari 
		meanvalue.append([mv0,mv1])
		variance.append([vari0,vari1,vari])
	# print meanvalue
	# print variance
	return meanvalue, variance, a_0, a_1

def test(DataSet, LabelSet, meanvalue, variance, a_0, a_1):
	predict = []
	acc = 0.0
	for x in range(len(DataSet)):
		p_1 = 1
		p_0 = 1
		PP0  = 0
		PP1  = 0
		for i in range(len(DataSet[0])):
			if variance[i][0] == 0.0:
				variance[i][0] = 0.0001
			if variance[i][1] == 0.0:
				variance[i][1] = 0.0001
			if variance[i][2] == 0.0:
				variance[i][2] = 0.0001
			p_0 =  stats.norm.pdf(DataSet[x][i], meanvalue[i][0], variance[i][0]) 
			p_1 =  stats.norm.pdf(DataSet[x][i], meanvalue[i][1], variance[i][1]) 
			p1  = stats.norm.pdf(DataSet[x][i], meanvalue[i][1], variance[i][2])
			p0  = stats.norm.pdf(DataSet[x][i], meanvalue[i][0], variance[i][2])
			PP0 += log(max(p_0, 1e-20))
			PP1 += log(max(p_1, 1e-20))
		if PP1 + log(a_1) >= PP0 + log(a_0):
			predict.append(1.0)
		else :
			predict.append(0.0)
		# print len(predict)
		if predict[-1] == LabelSet[x][0]:
			acc += 1
	return acc / len(DataSet)


TrainData = loadData('/Users/skylatitude/downloads/spam_polluted/train_feature.txt')
TrainLabel = loadData('/Users/skylatitude/downloads/spam_polluted/train_label.txt')
TestData = loadData('/Users/skylatitude/downloads/spam_polluted/test_feature.txt')
TestLabel = loadData('/Users/skylatitude/downloads/spam_polluted/test_label.txt')
trainnum = len(TrainData)
DataSet = TrainData[:]
print shape(TrainData)
print '############################################################'
DataSet.extend(TestData)
DataSetlowD = PCA(n_components=100)
Data = DataSetlowD.fit(DataSet).transform(DataSet)
Train = Data[0:trainnum]
Test = Data[trainnum:]
print shape(Train), shape(Test)
meanvalue, result, a_0, a_1 = Normal(Train,TrainLabel)
acc = test(Test, TestLabel, meanvalue, result, a_0, a_1)
print 'The ACC of test Folder is %f' %(acc)




