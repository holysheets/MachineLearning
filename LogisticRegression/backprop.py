#Theresa Sheets
#Code for Back Propagation

import argparse, sys, math, random, numpy
from scipy.optimize import minimize

parser= argparse.ArgumentParser()
parser.add_argument("--train", help="Gives the data for training")
parser.add_argument("--test", help="Gives the data for testing")
parser.add_argument("--r", help="Learning rate", type=float)
parser.add_argument("--d",help="help with convergence",type=float)
parser.add_argument("--width", help="width of the layers", type=int)
parser.add_argument("--T", help='number of epochs', type=int)
parser.add_argument("--zero", help='initialize weights to 0', action="store_true")
args = parser.parse_args()


def sigmoid(activation):
    return 1.0/(1.0+math.exp(-activation))

def sigmoid_derivative(output):
    return output*(1.0-output)

def activate(weights, row):
    activation=weights[-1]
    for i in range(len(weights)-1):
        #print(len(weights))
        activation=activation+weights[i]*row[i]
    return activation


def make_network():
    network=list()
    numpy.random.seed(23)
    width=args.width
    numLayer=2
    numOutput=2
    numInput=width
    if args.zero:
        firstLayer=[{'weights':[0 for i in range(4)]} for i in range(width)]
        hiddenLayers=[{'weights':[0 for i in range(width+1)]} for i in range(width)]
        outputLayer=[{'weights':[0 for i in range(width+1)]} for i in range(numOutput)]
    else:
        firstLayer=[{'weights':[numpy.random.normal(1) for i in range(4)]} for i in range(width)]
        hiddenLayers=[{'weights':[numpy.random.normal(1) for i in range(width+1)]} for i in range(width)]
        outputLayer=[{'weights':[numpy.random.normal(1)for i in range(width+1)]} for i in range(numOutput)]
    network.append(firstLayer)
    network.append(hiddenLayers)
    network.append(outputLayer)
    #print(outputLayer)
    return network


def predict(network, row):
    outputs=forward_prop(network,row)
    return outputs.index(max(outputs))

def forward_prop(network,row):
    inputs=row
    for layer in network:
        newData=[]
        for entry in layer:
            activation=activate(entry['weights'],inputs)
            entry['output']=sigmoid(activation)
            newData.append(entry['output'])
        inputs=newData
    return inputs


def back_prop(network,expected):
    networkLen=len(network)
    for i in reversed(range(networkLen)):
        layer=network[i]
        error=list()
        if i!=(networkLen-1):
            for j in range(len(layer)):
                err=0.0
                for entry in network[i+1]:
                    err=err+(entry['weights'][j]*entry['delta'])
                error.append(err)
        else:
            for j in range(len(layer)):
                entry=layer[j]
                error.append(expected[j]-entry['output'])
        for j in range(len(layer)):
            entry=layer[j]
            entry['delta']=error[j]*sigmoid_derivative(entry['output'])

def updateWeights(network, row):
    networkLen=len(network)
    learningRate=args.r
    d=args.d
    for i in range(networkLen):
        data=row[:-1]
        rate=learningRate/(1.0+learningRate*i/d)
        if i!=0:
            data=[entry['output'] for entry in network[i-1]]
        for entry in network[i]:
            for j in range(len(data)):
                entry['weights'][j]=entry['weights'][j]+rate*entry['delta']*data[j]
            entry['weights'][-1]=entry['weights'][-1]+rate*entry['delta']

def train_set(data):
    network=make_network()
    epochs=args.T
    numOutput=2
    for i in range(epochs):
        error_sum=0
        random.shuffle(data)
        for x in data:
            outputs=forward_prop(network,x)
            expected=[0 for i in range(numOutput)]
            expected[x[-1]]=1
            error_sum=error_sum+sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            back_prop(network, expected)
            updateWeights(network,x)
        print("Epoch "+str(i)+", error "+str(error_sum))
    return network



def test_set(data, network):
    dataLen=len(data)
    count=0
    for x in data:
        prediction=predict(network,x)
        if prediction==x[-1]:
            count=count+1
    error=count/float(dataLen)
    return error


#adds bias weight before the label
def fixData(data):
    for x in data:
        for i in range(len(x)-1):
            x[i]=float(x[i])
        x[-1]=int(x[-1])
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
            network=train_set(idx)
            error=test_set(testIdx,network)
            print('The error on the test data is: '+str(error))
            error=test_set(idx,network)
            print('The error on the training data is: '+str(error))
