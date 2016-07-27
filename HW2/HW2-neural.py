from numpy import *

def Inputdata():
    Data = identity(8)
    return Data

def sigmoid(z):
    return 1.0/(1 + exp(-z))

def MSE(predict, true):
    sum(predict - true)
    pass

def neural(Input):
    l = 0.5
    FirstInput = mat(random.rand(1, 8))
    FirstInput -= FirstInput
    Winput = mat(random.rand(3, 8))
    Woutput = mat(random.rand(8, 3))
    Itheta = random.rand(1, 3)
    Otheta = random.rand(1, 8)
    Sample = random.rand(8, 8)
    HiddenInput = HiddenOutput=[0] * 3
    LastInput = LastOutput = [0] * 8
    Err = [0] * 8
    err = [0] * 3
    iteration = 1
    while(1):
        for data in Input:
                #for j in range(len(input[i])):
            FirstInput = mat(random.rand(1, 8))
            FirstInput -= FirstInput
            FirstInput += data
            for j in range(3):
                HiddenInput[j] = Winput[j] * FirstInput.T + Itheta[0][j]
                HiddenOutput[j] = sigmoid(HiddenInput[j])
            for j in range(8):
                LastInput[j] = Woutput[j] * mat(array(HiddenOutput)).T + Otheta[0][j]
                LastOutput[j] = sigmoid(LastInput[j])
            for j in range(len(LastOutput)):
                Err[j] = (data[j] - LastOutput[j]) * LastOutput[j] * (1 - LastOutput[j])

            for j in range(len(HiddenOutput)):
                    #print shape(mat(array(errO))), shape(Woutput.T[j].T)
                err[j] = HiddenOutput[j] * (1 - HiddenOutput[j]) * (mat(array(Err)) * Woutput.T[j].T)

            for n in range(len(LastOutput)):
                    #print Woutput[n]
                for j in range(len(HiddenOutput)):
                    Woutput[n, j] += l * Err[n] * HiddenOutput[j]
                    #print Woutput[n]
                #break
            for n in range(len(HiddenOutput)):
                for j in range(8):
                    Winput[n, j] += l * err[n] * FirstInput[0, j]
            for j in range(len(LastOutput)):
                Otheta[0][j] += l * Err[j]
            for j in range(len(HiddenOutput)):
                Itheta[0][j] += l * err[j]
            
            #break
        MSE = 0.0
        for j in range(len(LastOutput)):
            MSE = MSE + (LastOutput[j] - data[j])**2
        iteration += 1
        if MSE < 0.01:
            break   

    for i in range(8):
        
        FirstInput = mat(random.rand(1, 8))
        FirstInput -= FirstInput
        FirstInput += Input[i]
        for j in range(3):
            HiddenInput[j] = Winput[j] * FirstInput.T + Itheta[0][j]
            HiddenOutput[j] = sigmoid(HiddenInput[j])
        for j in range(8):
            LastInput[j] = Woutput[j] * mat(array(HiddenOutput)).T + Otheta[0][j]
            LastOutput[j] = sigmoid(LastInput[j])
        for j in range(8):
            if LastOutput[j] < 0.1:
                Sample[i,j] = 0
            else:
                Sample[i,j] = 1

    print LastOutput
    print Sample


Data = Inputdata()
neural(Data)