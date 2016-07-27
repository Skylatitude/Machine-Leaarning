import os, struct
from array import array as pyarray
from numpy import *


def load_mnist(dataset="training", digits=arange(10), path="."):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [k for k in range(size) if lbl[k] in digits]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels


def featureGeneralization(images):
    featurelist = []
    for x in range(200):
        featurevec = []
        for i in range(len(images)):
            eara = 0
            while 130 > eara or eara > 170:
                x1 = random.randint(1, 27)
                y1 = random.randint(1, 27)
                x0 = random.randint(0, x1)
                y0 = random.randint(0, y1)
                eara = x1 * y1 - x0 * y1 - x1 * y0 + x0 * y0
            feature = sum(images[i][x0:x1][y0:y1])
            featurevec.append(feature)
        featurelist.append(featurevec)
    return transpose(featurelist)


def creatFunctionTable():
    table = mat(ones((10, 50)))
    tableinfo = []
    for i in range(50):
        f = []
        numofFeature = random.randint(0, 9)
        for j in range(numofFeature):
            t = random.randint(0, 9)
            table[t, i] = -1
            f.append(t)

        for n in range(i):
            if f == tableinfo[n]:
                m = random.randint(0, 9)
                table[m, i] = -1
                f.append(m)
                continue
        tableinfo.append(list(set(f)))
    return table, tableinfo


def translabels(labels, tableinfo):
    temp = []
    for i in range(len(labels)):
        if labels[i] in tableinfo:
            temp.append(-1)
        else:
            temp.append(1)
    return temp


def classify(DataSet, colum, threshold, mark):
    predictArray = ones((shape(DataSet)[0],1))
    if mark == 'small':
        predictArray[DataSet[:,colum] <= threshold] = -1
    else:
        predictArray[DataSet[:,colum] > threshold]  = -1
    return predictArray

def featureExtr(image, test):
    x1, x2, y1, y2 = [], [], [], []
    for i in range(100):
        area = 0
        while area < 130 or area > 170:
            a, b, c, d = random.randint(0, 27), random.randint(0, 27), random.randint(0, 27), random.randint(0, 27)
            area = abs((a - b) * (c - d))

        x1.append(a)
        x2.append(b)
        y1.append(c)
        y2.append(d)


        if x1[i] > x2[i]:
            (x1[i], x2[i]) = (x2[i], x1[i])

        if y1[i] > y2[i]:
            (y1[i], y2[i]) = (y2[i], y1[i])



    dataset = []
    for i in range(len(image)):
        feature = []
        for j in range(100):
            midx = (x2[j] - x1[j]) /2 + x1[j]
            midy = (y2[j] - y1[j]) /2 + y1[j]
            # print midx
            countx1 = int(sum(image[i][x1[j]:midx][y1[j]:y2[j]]))
            countx2 = int(sum(image[i][midx:x2[j]][y1[j]:y2[j]]))
            countx = countx1 - countx2
            county1 = int(sum(image[i][x1[j]:x2[j]][y1[j]:midy]))
            county2 =  int(sum(image[i][x1[j]:x2[j]][midy:y2[j]]))
            county = county1 - county2
            #print count
            feature.append(countx)
            feature.append(county)
        dataset.append(feature)
    testset = []
    for i in range(len(test)):
        feature = []
        for j in range(100):
            midx = (x2[j] - x1[j]) /2 + x1[j]
            midy = (y2[j] - y1[j]) /2 + y1[j]
            # print x1[j], midx, x2[j]
            # print midx
            countx1 = int(sum(test[i][x1[j]:midx][y1[j]:y2[j]]))
            countx2 = int(sum(test[i][midx:x2[j]][y1[j]:y2[j]]))
            countx = countx1 - countx2
            county1 = int(sum(test[i][x1[j]:x2[j]][y1[j]:midy]))
            county2 =  int(sum(test[i][x1[j]:x2[j]][midy:y2[j]]))
            county = county1 - county2
            #print count
            feature.append(countx)
            feature.append(county)
        testset.append(feature)
    return dataset, testset

def RandTree(traindata, trainlabel, weight):
    datamatrix = mat(traindata)
    labelmatrix = mat([trainlabel])
    m, n = shape(datamatrix)
    bestStump = {}
    bestClass = mat(zeros((m,1)))
    minErr = inf
    p = random.randint(0, m - 1)
    q = random.randint(0, n - 1)
    bestStump['datapoint'] = p
    bestStump['feature'] = q
    bestStump['thres'] = datamatrix[p,q]
    for where in ['small', 'large']:
        threshold = datamatrix[p, q]
        predict = classify(datamatrix, q, threshold, where)
        errArr = mat(ones((m,1)))
        errArr[predict == labelmatrix.T] = 0
        weighterr = weight.T * errArr
        if weighterr < minErr:
            minErr = weighterr
            bestClass = predict.copy()
            bestStump['chose'] = where
            # bestStump['list'] = shortlist
    return bestStump, minErr, bestClass


def buildTree(traindata, trainlabel, weight):
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
    return bestStump, minErr, bestClass

def adaboost(traindata, trainlabel):
    labelmat = mat(trainlabel)
    result = []
    datanum = len(traindata)
    weight = mat(ones((datanum, 1))) / datanum
    predict = mat(zeros((datanum, 1)))
    iteration = 0
    while 1:
        # errArr = []
        bestStump, error, bestClass = RandTree(traindata, trainlabel, weight)
        alpha = 0.5 * log((1.0 - error)/ max(error, 1e-10))
        bestStump['alpha'] = alpha
        result.append(bestStump)
        # for i in range(datanum):
        #     if bestClass[i] == trainlabel[i]:
        #         weight[i] = weight[i] * exp(-alpha)
        #     else:
        #         weight[i] = weight[i] * exp(alpha)
        # print type(bestClass)
        ex = multiply(-1 * alpha[0,0]*labelmat.T,bestClass)
        weight = multiply(weight, exp(ex))
        weight = weight / sum(weight)

        predict += alpha[0,0] * bestClass
        errate = sum((multiply(labelmat.T, sign(predict)) - mat(ones((len(traindata),1))))/-2)
        errate /= datanum

        # print shape(predict), shape(mat(trainlabel))
        # for i in range(datanum):
        #     if sign(predict[i]) == trainlabel[i]:
        #         errArr.append(0.0)
        #     else:
        #         errArr.append(1.0)
        # errat = sum(errArr) / datanum
        # print errat, errate
        iteration += 1
        # print iteration, errate, error[0, 0]
        # pre = adaboosttest(traindata, trainlabel, result)
        if errate == 0.0 or iteration >= 8000:
            print 'termination error rate : ',errate
            break
    return result

def adaboosttest(testdata, testlabel, result):
    testmatrix = mat(testdata)
    m, n = shape(testdata)
    err = 0.0
    testresult = mat(zeros((m, 1)))
    listresult = []
    for i in range(len(result)):
        pre = classify(testmatrix, result[i]['feature'], result[i]['thres'], result[i]['chose'])
        testresult += mat(pre) * result[i]['alpha']
    for i in range(m):
        if sign(testresult[i]) == testlabel[i]:
            listresult.append(testresult[i,0])
            err += 1.0
    # print 'The acc is %f' %(err/m)
    return sign(testresult)

def translateLabel(predict, testlabel, table):
    acc = 0
    for i in range(len(predict)):
        for j in range(len(table)):
            minErr = inf
            err = abs(table[j] - predict[i])
            Err = sum(err)
            if minErr > Err :
                minErr = Err
                targ = j
        if targ == testlabel[i]:
            acc += 1
    return acc

# images, labels = load_mnist('training', digits=[1])
images, labels = load_mnist("training")
test, tlabel = load_mnist("testing")
images /= 128.0
test /= 128.0
testlabel=labels[:3000]
# imshow(images.mean(axis=0), cmap=cm.gray)
# print shape(images)
# print shape(labels)
traindata, testdata = featureExtr(images[:3000], test[:3000])
# print traindata,'\n' ,testdata
# train = mat(zeros((200,1)))
# test = mat(zeros((200,1)))
# print shape(test)
# e = 0.0
# l = 0.0
# for i in range(len(testdata)):
#     if tlabel[i] == 1:
#         test += mat(testdata[i]).T
#         e += 1
#         # print testdata[i],'\n'
#         # break
# test /= e
# for i in range(len(traindata)):
#     if labels[i] == 1:
#         train += mat(traindata[i]).T
#         l += 1
#         # print traindata[i]
#         # break
# train /= l
# print 'Test : ', test, '\n', 'Train : ', train
table, tableinfo = creatFunctionTable()
pre = []
print shape(testdata)
for i in range(50):
    trainlabels = translabels(labels[:3000], tableinfo[i])
    testlabels = translabels(tlabel[:3000], tableinfo[i])
    result = adaboost(mat(traindata), trainlabels)
    predict = adaboosttest(testdata, testlabels, result)
    tem = adaboosttest(mat(traindata), trainlabels, result)
    # print shape(predict)
    pre.append(tem)
# print shape(pre)
# print pre
acc = 0.0
lentable = len(tlabel[:300])-1
for i in range(len(testlabel)):
    fe = []
    # print i
    for n in range(50):
        fe.append(pre[n][i][0,0])
    accate = 100
    for j in range(10):
        # print shape(table[j]),shape(fe)
        # print fe
        if sum(abs(table[j].T - mat(fe[:][:][0]))) < accate:
            accate = sum(abs(table[j].T - mat(fe)))
            min = j
    if min == int(testlabel[i]):
        acc += 1
print acc/lentable

