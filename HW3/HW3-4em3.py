from numpy import *
import math
import scipy.stats as stats
import numpy as np

def loadData(filename):
	vector = []
	with open(filename) as file:
		for line in file:
			vector.append(map(float,line.split()))
	return array(vector)

def prob(data, mean, matir):
	temp = mat(data - mean)
	sinv = linalg.pinv(matir)
	a = linalg.det(matir)
	if a == 0.0:
		a = 0.001
	k = (2.0*math.pi)**(-len(data)/2.0)*(1.0/(a) ** 0.5)
	p = k * exp(-0.5*(temp*sinv*temp.T))
	return p[0,0]

def init(DataSet):
	dataset1 = DataSet[0 :  3000]
	dataset2 = DataSet[3000:6000]
	dataset3 = DataSet[6000:    ]
	sigma1 = cov(transpose(dataset1))
	sigma2 = cov(transpose(dataset2))
	sigma3 = cov(transpose(dataset3))
	for data in dataset1:
		left1 = []
		right1 = []
		for i in range(len(DataSet[0])):
			if i == 0:
				left1.append(data[i])
			else:
				right1.append(data[i])
	mv1 = [mean(left1), mean(right1)]
	for data in dataset2:
		left2 = []
		right2 = []
		for i in range(len(DataSet[0])):
			if i == 0:
				left2.append(data[i])
			else:
				right2.append(data[i])
	mv2 = [mean(left2), mean(right2)]
	for data in dataset1:
		left3 = []
		right3 = []
		for i in range(len(DataSet[0])):
			if i == 0:
				left3.append(data[i])
			else:
				right3.append(data[i])
	mv3 = [mean(left3), mean(right3)]


	d1 = 0.3
	d2 = 0.3
	d3 = 0.4
	return array(mv1), array(mv2), array(mv3), sigma1, sigma2, sigma3, d1, d2, d3

def E_Step(mv1, mv2, mv3, sigma1, sigma2, sigma3, d1, d2, d3, DataSet):
	z1 = []
	z2 = []
	z3 = []
	for data in DataSet:
		p1 = prob(data, mv1, sigma1)
		p2 = prob(data, mv2, sigma2)
		p3 = prob(data, mv3, sigma3)
		t1 = d1 * p1/(p1*d1 + p2*d2 + p3*d3)
		t2 = d2 * p2/(p1*d1 + p2*d2 + p3*d3)
		t3 = d3 * p3/(p1*d1 + p2*d2 + p3*d3)
		z1.append(t1)
		z2.append(t2)
		z3.append(t3)
	return z1, z2, z3


def M_step(z1, z2, z3, dataset):
    sum1, sum2, sum3 = [0.0] * 2, [0.0] * 2, [0.0] * 2
    s_1 = sum(z1)
    s_2 = sum(z2)
    s_3 = sum(z3)
    for i in range(len(dataset)):    
        sum1 += dot(dataset[i], z1[i])
        sum2 += dot(dataset[i], z2[i])
        sum3 += dot(dataset[i], z3[i])
    d1 = s_1 / len(dataset)
    d2 = s_2 / len(dataset)
    d3 = s_3 / len(dataset)
    mv1 = sum1 / s_1
    mv2 = sum2 / s_2
    mv3 = sum3 / s_3
    sigma1 = cov(transpose(dataset))
    sigma1 -= sigma1
    sigma2 = cov(transpose(dataset))
    sigma2 -= sigma2
    sigma3 = cov(transpose(dataset))
    sigma3 -= sigma3
    for i in range(len(dataset)):
        sigma1 += (z1[i] * dot(transpose(mat((array(dataset[i]))) - mat(array(mv1))), (mat(array(dataset[i])) - mat(array(mv1)))))
        sigma2 += (z2[i] * dot(transpose(mat((array(dataset[i]))) - mat(array(mv2))), (mat(array(dataset[i])) - mat(array(mv2)))))
        sigma3 += (z3[i] * dot(transpose(mat((array(dataset[i]))) - mat(array(mv3))), (mat(array(dataset[i])) - mat(array(mv3)))))
    sigma1 = sigma1/s_1
    sigma2 = sigma2/s_2
    sigma3 = sigma3/s_3
    return array(mv1), array(mv2), array(mv3), sigma1, sigma2, sigma3, d1, d2, d3

DataSet = loadData('/Users/skylatitude/Desktop/3gaussian.txt')
mv1, mv2, mv3, sigma1, sigma2, sigma3, d1, d2, d3 = init(DataSet)
s = 0.0
print 'The Gaussian Discriminant Analysis ...'
while(1):
	llh = 1
	z1, z2, z3= E_Step(mv1, mv2, mv3, sigma1, sigma2, sigma3, d1, d2, d3, DataSet)
	mv1, mv2, mv3, sigma1, sigma2, sigma3, d1, d2, d3= M_step(z1, z2, z3, DataSet)
	for data in DataSet:
		llh += ((prob(data, mv1, sigma1) * d1) + (prob(data, mv2, sigma2) * d2) + (prob(data, mv3, sigma3) * d3))
	if llh - s <= 0.2:
		break
	s = llh
print 'mean_1 :', mv2
print 'mean_2 :', mv1
print 'mean_3 :', mv3
print 'pi_1   :', d2
print 'pi_2   :', d1
print 'pi_3   :', d3
print 'sigma1 :', sigma2
print 'sigma2 :', sigma1
print 'sigma3 :', sigma3


