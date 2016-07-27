from numpy import *
from math import log
import numpy as np
import matplotlib.pyplot as plt

class LoadData(object):

    def __init__(self, filename):
        self.filename = filename


    def datasets(self):
        datasets = np.genfromtxt(self.filename, dtype = None)
        dataset = []
        for i in range(len(datasets)):
            typo = list(datasets[i])                                                ## transform from tuple to list
            dataset.append(typo)
        return dataset



class Nomalized(object):

    def nomalized(self, dataset):
        for i in range(len(dataset[0])-1):
            featurelist = [example[i] for example in dataset]
            minimum = min(featurelist)
            for j in range(len(featurelist)):
                featurelist[j] -= minimum
            maximum = max(featurelist)
            for j in range(len(featurelist)):
                featurelist[j] = float(featurelist[j]) / maximum
            for j in range(len(dataset)):
                dataset[j][i] = featurelist[j]
        return dataset




class Linear(object):

    def __init__(self,dataset):
        self.dataset = dataset


    def matrixfeature(self):
        X = [[] * len(self.dataset[0])] * len(self.dataset)
        Y = [[] * len(self.dataset[0])] * len(self.dataset)
        for i in range(len(self.dataset)):
            X[i] = self.dataset[i][0:-1]
            Y[i] = (self.dataset[i][-1:])
        return mat(X), mat(Y)

    def optimalW(self, X, Y):
        Xt = transpose(X)
        W = linalg.inv(Xt * X) * Xt * Y
        return W

    def error(self, X, Y, W, testdataset):
        predict = X * W
        Testfeature = [[] * len(testdataset[0])] * len(testdataset)
        Test =  [[] * len(testdataset[0])] * len(testdataset)
        for i in range(len(testdataset)):
            Testfeature[i] = testdataset[i][0:-1]
            Test[i] = (testdataset[i][-1:])
        Y0 = mat(Test)
        X0 = mat(Testfeature)
        predict0 = X0 * W
        Dif = predict0 - Y0
        err = 0
        for i in range(len(testdataset)):
            err += Dif[i] * Dif[i]
        return float(err)/len(testdataset)





class LinearRidge_Re(Linear):

    def optimalW(self, X, Y, Lambda):
        Xt = transpose(X)
        N= len(Xt * X)
        I = eye(N,  k = 0)
        W = linalg.inv(Xt * X + Lambda * I) * Xt * Y
        return W

    def Wlenth(self, W):
        return np.linalg.norm(W, ord=None)

class Gradient(Linear):

    def optimalW(self, X, Y , Lambda, dataset):
        #Xt = transpose(X)
        m, n = shape(X)
        W = ones((n, 1))
        #print shape(W), shape(X)
        while(1):
            for j in range(n):
                for i in range(m):
                    W[j, 0] -= Lambda * (X[i] * W - Y[i]) * X[i, j]
            mse = self.error(X, Y, W, dataset)
            if mse < 29.5:
                break
            print mse
        return W


class Run(object):

    def LR_Gdescent(self):
        set = ('housing data')
        TrainData = LoadData('/Users/skylatitude/Desktop/train set.txt')
        train = TrainData.datasets()
        TestData = LoadData('/Users/skylatitude/Desktop/test set.txt')
        test = TestData.datasets()
        adddata = train
        adddata.extend(test)
        a = Nomalized()
        train = a.nomalized(train)
        adddata = adddata[:len(train)]
        b = Gradient(adddata)
        c, d = b.matrixfeature()
        e = b.optimalW(c, d, 0.01, train)
        f = b.error(c, d, e, test)
        print '%s MSE is %f' % (set[0], f)
 

a = Run()
x = a.LR_Gdescent()
