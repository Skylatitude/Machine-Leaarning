from numpy import *
import matplotlib.pyplot as plt
import random 


class Importdataset(object):

    def __init__(self, filename):
        self.filename = filename

    ## extract the dataset from file

    def datasets(self):
        datasets = np.genfromtxt(self.filename, dtype = None)
        dataset = []
        for i in range(len(datasets)):
            typo = list(datasets[i])                                                ## transform from tuple to list
            dataset.append(typo)
        return dataset


    ## extract the dataset from file

    def datasets2(self):
        dataset = []
        try:
            f = open(self.filename, 'r')
        except IOError:
            print "This file does not exist"
        for line in f:
            dataset.append(map(float, line.split()))
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










class Logistic_GA():


    def __init__(self, dataset):
        self.dataset = dataset


    def matrixfeature(self):
        X = [[] * len(self.dataset[0])] * len(self.dataset)
        Y = [[] * len(self.dataset[0])] * len(self.dataset)
        for i in range(len(self.dataset)):
            X[i] = self.dataset[i][0:-1]
            Y[i] = (self.dataset[i][-1:])
        return mat(X), mat(Y)


    def optimalW(self, X, Y, Lambda, dataset):
        m, n = shape(X)
        W = ones((n, 1))
        while(1):
            for j in range(n):
                for i in range(m):
                    Hw = 1.0/(1 + exp(- (X[i] * W)))
                    W[j, 0] += Lambda * (Y[i] - Hw) * X[i, j]
            err = self.accg(X, Y, W, dataset)
            if err < 0.122:
                break
            print err
        return W


    def confusiontest(self, X, Y, W, testdataset, thre):
        Testfeature = [[] * len(testdataset[0])] * len(testdataset)
        Test =  [[] * len(testdataset[0])] * len(testdataset)
        for i in range(len(testdataset)):
            Testfeature[i] = testdataset[i][0:-1]
            Test[i] = (testdataset[i][-1:])
        Y0 = mat(Test)
        X0 = mat(Testfeature)
        predict0 = X0 * W
        PT, PF, NT, NF = 0, 0, 0, 0
        for i in range(len(testdataset)):
            predict0[i] = self.sigmoid(predict0[i])
            if predict0[i] >= thre and Y0[i] == 1:
                PT += 1
            if predict0[i] >= thre and Y0[i] == 0:
                PF += 1
            if predict0[i] < thre and Y0[i] == 1:
                NF += 1
            if predict0[i] < thre and Y0[i] == 0:
                NT += 1
        return PT, PF, NT, NF

    def sigmoid(self, z):
        return 1.0/(1 + exp(-z))


    def accg(self, X, Y, W, testdataset):
        Testfeature = [[] * len(testdataset[0])] * len(testdataset)
        Test =  [[] * len(testdataset[0])] * len(testdataset)
        for i in range(len(testdataset)):
            Testfeature[i] = testdataset[i][0:-1]
            Test[i] = (testdataset[i][-1:])
        Y0 = mat(Test)
        X0 = mat(Testfeature)
        predict0 = X0 * W
        mistake = 0.0
        for i in range(len(testdataset)):
            predict0[i] = self.sigmoid(predict0[i])
            if (predict0[i] >= 0.5 and Y0[i] == 0) or (predict0[i] < 0.5 and Y0[i] == 1):
                mistake += 1
        return mistake/len(testdataset)





class Run(object):


    def Logistic_ga(self):
        set = ('housing data', 'spambase data')
        a1 = Nomalized()
        a2 = Importdataset('/Users/skylatitude/Desktop/spambase/spambase.data')
        testdataset = a2.datasets2()
        testdataset = a1.nomalized(testdataset)
        dataset0 = testdataset
        random.shuffle(dataset0)


        for i in range(10):
            dataset1 = dataset0[0:len(dataset0)/10 * i]
            dataset1.extend(dataset0[len(dataset0)/10 * (i+1):-1])
            dataset2 = dataset0[len(dataset0)/10 * i:len(dataset0)/10 * (i+1)]
            b0 = Logistic_GA(dataset1)
            c0, d0 = b0.matrixfeature()
            e0 = b0.optimalW(c0, d0, 0.5, dataset1)
            f0 = b0.accg(c0, d0, e0, dataset2)
            print '%s folder %d ACC is %f' % (set[1], i+1, (1-f0))

    def plot(self):
        a1 = Nomalized()
        a2 = Importdataset('/Users/skylatitude/Desktop/spambase/spambase.data')
        testdataset = a2.datasets2()
        testdataset = a1.nomalized(testdataset)
        dataset0 = testdataset
        random.shuffle(dataset0)



        for i in range(10):
            dataset1 = dataset0[0:len(dataset0)/10 * i]
            dataset1.extend(dataset0[len(dataset0)/10 * (i+1):-1])
            dataset2 = dataset0[len(dataset0)/10 * i:len(dataset0)/10 * (i+1)]
            b0 = Logistic_GA(dataset1)
            c0, d0 = b0.matrixfeature()
            e0 = b0.optimalW(c0, d0, 0.5, dataset2)
            #f0 = b0.acc(c0, d0, e0, dataset2)
            #print '%s folder %d ACC is %f' % (set[1], i+1, (1-f0))
            #PT, PF, NT, NF = b0.confusion(c0, d0, e0, dataset2)
            #print "PT", PT, "PF", PF, "NT", NT, "NF", NF
            #break
            thre = 0.00
            PTlog = [0] * 2000
            PFlog = [0] * 2000
            NTlog = [0] * 2000
            NFlog = [0] * 2000
            for i in range(2000):
                thre += 0.0005
                PTlog[i], PFlog[i], NTlog[i], NFlog[i] = b0.confusiontest(c0, d0, e0, dataset2, thre)
            #print "PT", PT, "PF", PF, "NT", NT, "NF", NF
            break

        datalogx = []
        datalogy = []

        for i in range(2000):
            datalogy.append(float(PTlog[i]) / (PTlog[i] + NFlog[i]))
            datalogx.append(float(PFlog[i]) / (PFlog[i] + NTlog[i]))


        # datax = []
        # datay = []

        # for i in range(100):
        #     datay.append(float(PT[i]) / (PT[i] + NF[i]))
        #     datax.append(float(PF[i]) / (PF[i] + NT[i]))

        for i in range(2000):
            # plt.plot(datax[i], datay[i], 'yo-')
            plt.plot(datalogx[i], datalogy[i], 'b*')
            plt.plot(datalogx, datalogy)
        AUC = trapz(datalogy, datalogx)
        print AUC
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        plt.title('Logistic curve')
        plt.show()






a = Run()
b = a.plot()


