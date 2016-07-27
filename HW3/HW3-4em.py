from numpy import *
import math

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
	dataset1 = DataSet[0: 3000]
	dataset2 = DataSet[3000:  ]

	sigma1 = cov(transpose(dataset1))
	sigma2 = cov(transpose(dataset2))
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


	d1 = 0.5
	d2 = 0.5
	return array(mv1), array(mv2), sigma1, sigma2, d1, d2

def E_Step(mv1, mv2, sigma1, sigma2, d1, d2, DataSet):
	z1 = []
	z2 = []
	for data in DataSet:
		p1 = prob(data, mv1, sigma1)
		p2 = prob(data, mv2, sigma2)
		t1 = d1 * p1/(p1*d1 + p2*d2)
		t2 = d2 * p2/(p1*d1 + p2*d2)
		z1.append(t1)
		z2.append(t2)
	return z1, z2


def M_step(z1, z2, dataset):
    sum1, sum2 = 0.0, 0.0
    sumx1, sumx2 = [0.0] * 2, [0.0] * 2
    for i in range(len(dataset)):
        sum1 += z1[i]
        sum2 += z2[i]
        sumx1 += dot(dataset[i], z1[i])
        sumx2 += dot(dataset[i], z2[i])
    d1 = sum1 / len(dataset)
    d2 = sum2 / len(dataset)
    mean1 = sumx1 / sum1
    mean2 = sumx2 / sum2
    sigma1 = cov(transpose(dataset))
    sigma1 -= sigma1
    sigma2 = cov(transpose(dataset))
    sigma2 -= sigma2
    for i in range(len(dataset)):
        sigma1 += (z1[i] * dot(transpose(mat((array(dataset[i]))) - mat(array(mean1))), (mat(array(dataset[i])) - mat(array(mean1)))))
        sigma2 += (z2[i] * dot(transpose(mat((array(dataset[i]))) - mat(array(mean2))), (mat(array(dataset[i])) - mat(array(mean2)))))

    sigma1 /= sum1
    sigma2 /= sum2
    return array(mean1), array(mean2), sigma1, sigma2, d1, d2
# def M_Step(z1, z2, DataSet):
# 	sum1 = [0.0] * 2
# 	sum2 = [0.0] * 2
# 	for i in range(len(DataSet)):
# 		#print z1[i]
# 		sum1 += dot(DataSet[i], z1[i])
# 		sum2 += dot(DataSet[i], z2[i])
# 	mv1 = true_divide(sum1 , sum(z1))
# 	mv2 = true_divide(sum2 , sum(z2))
# 	print mv1, mv2
# 	d1 = sum(z1) / len(DataSet)
# 	d2 = sum(z2) / len(DataSet)
# 	sigma1 = cov(transpose(DataSet))
# 	sigma1 -= sigma1
# 	sigma2 = cov(transpose(DataSet))
# 	sigma2 -= sigma2
# 	for i in range(len(DataSet)):
# 		sigma1 += (z1[i] * dot(transpose(DataSet[i] - mv1), (DataSet[i] - mv1)))
# 		sigma2 += (z2[i] * dot(transpose(DataSet[i] - mv2), (DataSet[i] - mv2)))
# 	sigma1 = sigma1 / sum(z1)
# 	sigma2 = sigma2 / sum(z2)
# 	return array(mv1), array(mv2), sigma1, sigma2, d1, d2,




DataSet = loadData('/Users/skylatitude/Desktop/2gaussian.txt')
mv1, mv2, sigma1, sigma2, d1, d2 = init(DataSet)
s = 0.0
print 'The Gaussian Discriminant Analysis ...'
while(1):
	llh = 1
	z1, z2 = E_Step(mv1, mv2, sigma1, sigma2, d1, d2, DataSet)
	mv1, mv2, sigma1, sigma2, d1, d2 = M_step(z1, z2, DataSet)
	for data in DataSet:
		llh += ((prob(data, mv1, sigma1) * d1) + (prob(data, mv2, sigma2) * d2))
	if llh - s <= 0.1:
		break
	s = llh
print 'mean_1 :', mv2
print 'mean_2 :', mv1
print 'pi_1   :', d2
print 'pi_2   :', d1
print 'sigma1 :', sigma2
print 'sigma2 :', sigma1
