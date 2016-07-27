from numpy import *

def loadData(filename):
	vector = []
	with open(filename) as file:
		for line in file:
			vector.append(map(float,line.split()))
	return vector

def ProcessData(DataSet):
	for vector in DataSet:
		vector.insert(0, 1)
		if vector[-1] == -1:
			for i in range(len(vector[:-1])):
				vector[i] = vector[i] * (-1)	
	Data = []
	Label = []
	for data in DataSet:
		Label.append(data[-1])
		data[-1] = 1
		Data.append(data[:-1])
	return Data, Label

def CalcuTheData(data):
	# print shape(data[0])
	w = ones((len(data[0]),1))
	# print shape(w)
	data = mat(data)
	# print data
	k = 0
	while(1):

		M = 0
		for i in range(len(data)):
			J = data[i] * w
			# print shape(data[i])
			# print shape(w)
			# print J
			if J <= 0:
				w = w + 2*data[i].T
				M += 1
		k += 1
		print 'Iteration %d, total mistake %d' %(k, M)
		if M == 0:
			break
	return w




data = loadData('/Users/skylatitude/Desktop/per.txt')
data,label = ProcessData(data)
w = CalcuTheData(data)
print w