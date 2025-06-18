#Theresa Sheets Implimentation of the Perceptron ALgorithm

import argparse, sys, math, random, numpy

parser= argparse.ArgumentParser()
parser.add_argument("--train", help="Gives the data for training")
parser.add_argument("--test", help="Gives the data for testing")
parser.add_argument("--r", help="Learning rate", type=float)
parser.add_argument("--T", help="Max number of Epochs", type=int)
parser.add_argument("--voted", help="voted Perceptron",action="store_true")
parser.add_argument("--average", help="average Perceptron", action="store_true")
args = parser.parse_args()


#computes a dot product of two vectors where x is one longer than w
def dotProduct(w,x):
    if len(w)>len(x):
        print("dotproduct mismatch")
    dot=0
    for j in range(len(w)):
        dot+=w[j]*float(x[j])
    return dot

#updates weight
def updateWeight(weight,x,r):
    for i in range(len(weight)):
        weight[i]+=r*float(x[-1])*float(x[i])
    return weight

def train_set(data):
    #import pdb; pdb.set_trace()
    weight=[0 for x in range(0,len(data[1])-1)]
    C=[]
    weights=[]
    a=[0 for x in range(0,len(weight))]
    r=args.r
    T=args.T
    i=0
    if args.voted:
        #voted
        c=0
        while i<T:
            for x in data:
                prediction=float(x[-1])*dotProduct(weight,x)
                if (prediction<=0):
                    C.append(c)
                    weights.append(weight)
                    print("Weight: "+str(weight))
                    print("The count for this weight is: "+str(c)+"\n")
                    weight=updateWeight(weight,x,r)
                    c=0
                c+=1
            i+=1
        return (weights, C)
    elif args.average:
        while i<T:
            for x in data:
                prediction=float(x[-1])*dotProduct(weight,x)
                if (prediction<=0):
                    weight=updateWeight(weight,x,r)
                for j in range(len(a)):
                    a[j]+=weight[j]

                #print(a)
            i+=1
            #print(i)
        return(a,C)
    else:
        while i<T:
            random.shuffle(data)
            for x in data:
                prediction=float(x[-1])*dotProduct(weight, x)
                if (prediction<=0):
                    #print("updating weight")
                    weight=updateWeight(weight,x,r)
            i+=1
            print(weight)
        return(weight,C)

def test_set(data, weight,C):
    dataLen=len(data)
    count=0
    if args.voted:
        prediction=0
        for x in data:
            for i in range(len(C)):
                label=float(x[-1])
                prediction+=float(C[i])*numpy.sign(dotProduct(weight[i],x))
            if prediction>0:
                prediction=1
            else:
                prediction=-1
            if prediction==label:
                count+=1
    else:
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
            x.append(-1.0)
        if label==1:
            x.append(1.0)
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
            (weight,C)=train_set(idx)
            error=test_set(testIdx,weight,C)
            if not args.voted:
                print('The weight vector is: '+str(weight))
            print('The error on the test data is: '+str(error))
            error=test_set(idx,weight,C)
            print('The error on the training data is: '+str(error))
