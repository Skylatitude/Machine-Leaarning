from numpy import *
import random


def loadData(filename):
	vector = []
	with open(filename) as file:

		for line in file:
			vector.append(map(float,line.split()))
	return vector

def ShuffleData(DataSet):
	
	random.shuffle(DataSet)
	return DataSet

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
		Featurelist = getfeature(DataSet, i)
		he = max(Featurelist) - min(Featurelist)
		h1 = he / 9
		p1_0 = 0.0
		p2_0 = 0.0
		p3_0 = 0.0
		p4_0 = 0.0
		p5_0 = 0.0
		p6_0 = 0.0
		p7_0 = 0.0
		p8_0 = 0.0
		p9_0 = 0.0
		p1_1 = 0.0
		p2_1 = 0.0
		p3_1 = 0.0
		p4_1 = 0.0
		p5_1 = 0.0
		p6_1 = 0.0
		p7_1 = 0.0
		p8_1 = 0.0
		p9_1 = 0.0
		for data in DataSet:
			if data[-1] == 0:
				if data[i] <= h1:
					p1_0 += 1
				elif data[i] <= h1*2:
					p2_0 += 1
				elif data[i] <= h1*3:
					p3_0 += 1
				elif data[i] <= h1*4:
					p4_0 += 1
				elif data[i] <= h1*5:
					p5_0 += 1
				elif data[i] <= h1*6:
					p6_0 += 1
				elif data[i] <= h1*7:
					p7_0 += 1
				elif data[i] <= h1*8:
					p8_0 += 1
				else: 
					p9_0 += 1
				a_0 += 1
			if data[-1] == 1:
				if data[i] <= h1:
					p1_1 += 1
				elif data[i] <= h1*2:
					p2_1 += 1
				elif data[i] <= h1*3:
					p3_1 += 1
				elif data[i] <= h1*4:
					p4_1 += 1
				elif data[i] <= h1*5:
					p5_1 += 1
				elif data[i] <= h1*6:
					p6_1 += 1
				elif data[i] <= h1*7:
					p7_1 += 1
				elif data[i] <= h1*8:
					p8_1 += 1
				else:
					p9_1 += 1
				a_1 += 1
		meanvalue.append([h1, h1*2, h1*3, h1*4, h1*5, h1*6, h1*7, h1*8])
		p1_0 = p1_0 / DataNum
		p2_0 = p2_0 / DataNum
		p3_0 = p3_0 / DataNum
		p4_0 = p4_0 / DataNum
		p5_0 = p5_0 / DataNum
		p6_0 = p6_0 / DataNum
		p7_0 = p7_0 / DataNum
		p8_0 = p8_0 / DataNum
		p9_0 = p9_0 / DataNum
		p1_1 = p1_1 / DataNum
		p2_1 = p2_1 / DataNum
		p3_1 = p3_1 / DataNum
		p4_1 = p4_1 / DataNum
		p5_1 = p5_1 / DataNum
		p6_1 = p6_1 / DataNum
		p7_1 = p7_1 / DataNum
		p8_1 = p8_1 / DataNum
		p9_1 = p9_1 / DataNum
		result.append([p1_0, p2_0, p3_0, p4_0, p5_0, p6_0, p7_0, p8_0, p9_0, p1_1, p2_1, p3_1, p4_1, p5_1, p6_1, p7_1, p8_1, p9_1])
	return meanvalue ,result , a_0, a_1

def test(DataSet, meanvalue, result, a_0, a_1):
	predict = []
	acc =0.0
	for data in DataSet:
		p_1 = 1
		p_0 = 1
		for i in range(len(DataSet[0])-1):
			if data[i] <= meanvalue[i][0]:
				p_1 = p_1 * result[i][9] * a_0
				p_0 = p_0 * result[i][0] * a_1
			elif data[i] <= meanvalue[i][1]:
				p_1 = result[i][10] * p_1 * a_0
				p_0 = result[i][1] * p_0 * a_1
			elif data[i] <= meanvalue[i][2]:
				p_1 = result[i][11] * p_1 * a_0
				p_0 = result[i][2] * p_0 * a_1
			elif data[i] <= meanvalue[i][2]:
				p_1 = result[i][12] * p_1 * a_0
				p_0 = result[i][3] * p_0 * a_1
			elif data[i] <= meanvalue[i][2]:
				p_1 = result[i][13] * p_1 * a_0
				p_0 = result[i][4] * p_0 * a_1
			elif data[i] <= meanvalue[i][2]:
				p_1 = result[i][14] * p_1 * a_0
				p_0 = result[i][5] * p_0 * a_1
			elif data[i] <= meanvalue[i][2]:
				p_1 = result[i][15] * p_1 * a_0
				p_0 = result[i][6] * p_0 * a_1
			elif data[i] <= meanvalue[i][2]:
				p_1 = result[i][16] * p_1 * a_0
				p_0 = result[i][7] * p_0 * a_1
			else:
				p_1 = result[i][17] * p_1 * a_0
				p_0 = result[i][8] * p_0 * a_1
		if p_1 > p_0:
			predict.append(1)

		else :
			predict.append(0)
		if data[-1] == predict[-1]:
			acc += 1
	return acc / len(DataSet)

DataSet = loadData('/Users/skylatitude/Desktop/spambase/spambase.data')
DataSet = ShuffleData(DataSet)
SetNum = (len(DataSet) -1 ) / 10
for i in range(10):
	TestData = DataSet[SetNum*i:(SetNum*i + SetNum)]
	TrainData = DataSet[0: SetNum*i]
	TrainData.extend(DataSet[(SetNum*i + SetNum):-1])
	meanvalue, result, a_0, a_1 = his(TrainData)
	acc = test(TestData,meanvalue, result, a_0, a_1)
	print 'The ACC of %d Folder is %f' %(i+1,acc)
