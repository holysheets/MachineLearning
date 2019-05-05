#Theresa Sheets Implimentation of the Perceptron ALgorithm
#CMSC 6350 HW 4
#April 5, 2019


import argparse, sys, math, random, numpy
from scipy.optimize import minimize

parser= argparse.ArgumentParser()
parser.add_argument("--train", help="Gives the data for training")
parser.add_argument("--test", help="Gives the data for testing")
parser.add_argument("--r", help="Learning rate", type=float)
parser.add_argument("--T", help="Max number of Epochs", type=int)
parser.add_argument("--C", help="C", type=float)
parser.add_argument("--rateFunct", help="Which learning rate schedule", type=int)
parser.add_argument("--d",help="help with convergence",type=float)
parser.add_argument("--duel", help="Use the duel form",action='store_true')
args = parser.parse_args()


#computes a dot product of two vectors where x is one longer than w
def dotProduct(w,x):
    if len(w)>len(x):
        print("dotproduct mismatch")
    dot=0
    for j in range(len(w)):
        dot+=w[j]*float(x[j])
    return dot


def train_set_sgd(data):
    #import pdb; pdb.set_trace()
    weight=[0 for x in range(0,len(data[1])-1)]
    T=args.T
    C=args.C
    r=args.r
    i=0

    while i<T:
        random.shuffle(data)
        m=0
        objective=[]
        for x in data:
            if args.rateFunct==1:
                rate=r/(1+r*m/args.d)
            if args.rateFunct==2:
                rate=r/(1+m)
            value=x[-1]*dotProduct(weight,x)
            if value<=1:
                for j in range(len(weight)):
                    weight[j]=(1-rate)*float(weight[j])+rate*C*x[-1]*float(x[j])
                #print("if")
            else:
                #print("else")
                for j in range(len(weight)):
                    weight[j]=(1-rate)*float(weight[j])
            objective.append(loss_f(data,weight,C))
            #if (1-value)<0:
            #    value=0
            #objective.append(0.5*dotProduct(weight, weight)+C*value)
            m=m+1
        i=i+1
        #print(objective)
    return weight

def loss_f(data,weight,C):
    objective=0
    for x in data:
        value=1-x[-1]*dotProduct(weight,x)
        if value>0:
            objective=objective+value
    return (0.5*dotProduct(weight, weight)+C*objective)



def train_set_duel(data):
    dataLen=len(data)
    sampleLength=len(data[0])-1
    C=args.C
    a0=numpy.zeros(dataLen)
    boundaries=tuple([(0,C) for i in range(dataLen)])
    labels=[m[-1] for m in data]

    K_mat=numpy.ndarray([dataLen,dataLen])
    for i in range(dataLen):
        for j in range(dataLen):
            #import pdb; pdb.set_trace()
            K_mat[i,j]=(data[i][-1])*(data[j][-1])*numpy.inner(data[i][0:sampleLength],data[j][0:sampleLength])



    def constr(x):
        return numpy.inner(x,numpy.asarray(labels))

    constrain={'type':'eq','fun':constr}

    def objectiveF(x):
        return 0.5*numpy.dot(x.T, numpy.dot(K_mat,x))-sum(x)

    sol=minimize(objectiveF,a0,method='SLSQP', bounds= boundaries, constraints=constrain)

    def get_weight(solution):
        length=len(solution)
        weights=[]
        for i in range(length):
            weights.append(solution[i]*data[i][-1]*numpy.asarray(data[i][0:sampleLength]))
        return sum(weights)

    weight=get_weight(sol.x)






    return weight


def test_set(data, weight):
    dataLen=len(data)
    count=0
    for x in data:
        label=float(x[-1])
        prediction=dotProduct(weight,x)
        if prediction>0:
            prediction=1
        else:
            prediction=-1
        if prediction==label:
            count+=1
    error=count/float(dataLen)
    return error


#adds bias weight before the label
def fixData(data):
    for x in data:
        label=float(x[-1])
        x[-1]=1
        if label==0:
            #print('zero')
            x.append(-1.0)
        if label==1:
            #print('one')
            x.append(1.0)
        for i in range(len(x)):
            x[i]=float(x[i])
    #import pdb; pdb.set_trace()
    return data


def load_file( datFile):
    idx=[]
    with open( datFile,'r') as inFile:
        for line in inFile:
            terms=line.strip().split(',')
            idx.append(terms)
    return idx

if __name__ == "__main__":
    if not args.train:
        print("ERROR: data files were not supplied for training")
        sys.exit(-1)
    else:
        if not args.test:
            print("No test files included")
            sys.exit(-1)
        else:
            idx=load_file(args.train)
            fixData(idx)
            testIdx=load_file(args.test)
            fixData(testIdx)
            if args.duel:
                weight=train_set_duel(idx)
            else:
                weight=train_set_sgd(idx)
            print('The weight is: '+str(weight))
            error=test_set(testIdx,weight)
            print('The error on the test data is: '+str(error))
            error=test_set(idx,weight)
            print('The error on the training data is: '+str(error))