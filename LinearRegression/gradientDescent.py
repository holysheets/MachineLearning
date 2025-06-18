#Theresa Sheets Gradient Descent Algorithm

import argparse, sys, math, random

parser= argparse.ArgumentParser()
parser.add_argument("--train", help="Gives the data for training")
parser.add_argument("--test", help="Gives the data for testing")
parser.add_argument("--r", help="Learning rate", type=float)
parser.add_argument("--stochastic", help="Adds stochasticity",action="store_true")
args = parser.parse_args()


#computes a dot product of two vectors where x is one longer than w
def dotProduct(w,x):
    if len(w)>len(x):
        print("dotproduct mismatch")
    dot=0
    for j in range(len(w)):
        dot+=w[j]*float(x[j])
    return dot


#finds the cost function
def cost(data,w,b):
    cost=0
    for x in data:
        cost+=.5*(float(x[-1])-dotProduct(w,x)-b)**2
    return cost

#finds gradient of W
def gradW(data,w,b):
    grad=[0 for x in range(0,len(w))]
    for j in range(len(w)):
        for x in data:
            grad[j]-=(float(x[-1])-dotProduct(w,x)-b)*float(x[j])
    return grad

#finds the gradient of B
def gradB(data, w, b):
    grad=0
    for x in data:
        grad-=float(x[-1])-dotProduct(w,x)-b
    return grad




def batch_update(data,r,w,b):
    newW=[0 for x in range(0, len(w))]
    grad=gradW(data,w,b)
    for j in range(len(w)):
        newW[j]=w[j]-r*grad[j]
    newB=b-r*gradB(data,w,b)
    return(newW, newB)

def stochastic_update(x, r, w, b):
    newW=[0 for j in range(0,len(w))]
    for j in range(len(w)):
        #import pdb; pdb.set_trace()
        newW[j]=w[j]+r*(float(x[-1])-dotProduct(w,x)-b)*float(x[j])
    newB=b+r*(float(x[-1])-dotProduct(w,x)-b)

    return(newW, newB)




def train_set(data,testData):
    weight=[0 for x in range(0,len(data[1])-1)]
    b=0
    r=args.r
    weightErr=1
    costErr=1
    iteration=0
    costInTime=[]
    testCostInTime=[]
    if not args.stochastic:
        #calculate boost Gradient
        while ((weightErr>(1e-6)) and (iteration<=1000)):
            (newW, newB)=batch_update(data,r,weight,b)
            difference=[0 for x in range(0,len(weight))]
            for j in range(len(weight)):
                difference[j]=weight[j]-newW[j]
            weightErr=dotProduct(difference,difference)
            weight=newW
            b=newB
            currentCost=cost(data,weight,b)
            testCurrentCost=cost(testData,weight,b)
            costInTime.append(currentCost)
            testCostInTime.append(testCurrentCost)
            iteration+=1
    else:
        #calculate stochastic Gradient
        while((costErr>(1e-6)) and (iteration<=1000)):
            random.shuffle(data)
            for x in data:
                (newW, newB)=stochastic_update(x, r, weight, b)
                costErr=abs(cost(data,weight,b)-cost(data,newW,newB))
                weight=newW
                b=newB
                currentCost=cost(data,weight,b)
                testCurrentCost=cost(testData,weight, b)
                testCostInTime.append(testCurrentCost)
                costInTime.append(currentCost)
                if costErr<(1e-6):
                    break
                iteration+=1
    print(testCostInTime)
    print('Number of iterations: '+str(iteration))
    return(weight, b)



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
            testIdx=load_file(args.test)
            (weight, b)=train_set(idx,testIdx)
            trainCost=cost(idx,weight,b)
            print('The cost of the training data is: '+str(trainCost))
            print('The weight vector is: '+str(weight))
            print('The bias is: '+str(b))
            testCost=cost(testIdx,weight,b)
            print('The cost of the test data is: '+str(testCost))
