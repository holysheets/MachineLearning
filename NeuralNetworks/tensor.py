#Theresa Sheets
#TensorFlow Code


import argparse, sys, math, random, numpy as np, tensorflow as tf
from tensorflow import keras
#from scipy.optimize import minimize

parser= argparse.ArgumentParser()
parser.add_argument("--train", help="Gives the data for training")
parser.add_argument("--test", help="Gives the data for testing")
parser.add_argument("--r", help="Learning rate", type=float)
parser.add_argument("--d",help="help with convergence",type=float)
parser.add_argument("--width", help="width of the layers", type=int)
parser.add_argument("--T", help='number of epochs', type=int)
parser.add_argument("--zero", help='initialize weights to 0', action="store_true")
parser.add_argument("--RELU", help="RELU", action="store_true")
parser.add_argument("--tanh", help="tanh", action="store_true")
parser.add_argument("--depth", help='depth', type=int)
args = parser.parse_args()


def train_set(data):
    dataUnlabeled=[np.array(x[0:(len(data[0])-1)], ndmin=2) for x in data]
    dataLabels=np.array([x[-1] for x in data])
    width=args.width
    numOutput=2
    epochs=args.T
    if args.RELU:
        if args.depth==3:
            network=keras.Sequential([keras.layers.Flatten(input_shape=(1,5)),
                keras.layers.Dense(width, activation=tf.nn.relu, kernel_initializer='he_normal', bias_initializer='zeros'),
                keras.layers.Dense(width, activation=tf.nn.relu, kernel_initializer='he_normal', bias_initializer='zeros'),
                keras.layers.Dense(numOutput, activation=tf.nn.softmax)])

        if args.depth==5:
            network=keras.Sequential([keras.layers.Flatten(input_shape=(1,5)),
                keras.layers.Dense(width, activation=tf.nn.relu, kernel_initializer='he_normal', bias_initializer='zeros'),
                keras.layers.Dense(width, activation=tf.nn.relu, kernel_initializer='he_normal', bias_initializer='zeros'),
                keras.layers.Dense(width, activation=tf.nn.relu, kernel_initializer='he_normal', bias_initializer='zeros'),
                keras.layers.Dense(width, activation=tf.nn.relu, kernel_initializer='he_normal', bias_initializer='zeros'),
                keras.layers.Dense(numOutput, activation=tf.nn.softmax)])

        if args.depth==9:
            network=keras.Sequential([keras.layers.Flatten(input_shape=(1,5)),
                keras.layers.Dense(width, activation=tf.nn.relu, kernel_initializer='he_normal', bias_initializer='zeros'),
                keras.layers.Dense(width, activation=tf.nn.relu, kernel_initializer='he_normal', bias_initializer='zeros'),
                keras.layers.Dense(width, activation=tf.nn.relu, kernel_initializer='he_normal', bias_initializer='zeros'),
                keras.layers.Dense(width, activation=tf.nn.relu, kernel_initializer='he_normal', bias_initializer='zeros'),
                keras.layers.Dense(width, activation=tf.nn.relu, kernel_initializer='he_normal', bias_initializer='zeros'),
                keras.layers.Dense(width, activation=tf.nn.relu, kernel_initializer='he_normal', bias_initializer='zeros'),
                keras.layers.Dense(width, activation=tf.nn.relu, kernel_initializer='he_normal', bias_initializer='zeros'),
                keras.layers.Dense(width, activation=tf.nn.relu, kernel_initializer='he_normal', bias_initializer='zeros'),
                keras.layers.Dense(numOutput, activation=tf.nn.softmax)])


    if args.tanh:
        if args.depth==3:
            network=keras.Sequential([keras.layers.Flatten(input_shape=(1,5)),
                keras.layers.Dense(width, activation=tf.nn.tanh, kernel_initializer='glorot_normal', bias_initializer='zeros'),
                keras.layers.Dense(width, activation=tf.nn.relu, kernel_initializer='glorot_normal', bias_initializer='zeros'),
                keras.layers.Dense(numOutput, activation=tf.nn.softmax)])

        if args.depth==5:
            network=keras.Sequential([keras.layers.Flatten(input_shape=(1,5)),
                keras.layers.Dense(width, activation=tf.nn.tanh, kernel_initializer='glorot_normal', bias_initializer='zeros'),
                keras.layers.Dense(width, activation=tf.nn.relu, kernel_initializer='glorot_normal', bias_initializer='zeros'),
                keras.layers.Dense(width, activation=tf.nn.relu, kernel_initializer='glorot_normal', bias_initializer='zeros'),
                keras.layers.Dense(width, activation=tf.nn.relu, kernel_initializer='glorot_normal', bias_initializer='zeros'),
                keras.layers.Dense(numOutput, activation=tf.nn.softmax)])

        if args.depth==9:
            network=keras.Sequential([keras.layers.Flatten(input_shape=(1,5)),
                keras.layers.Dense(width, activation=tf.nn.tanh, kernel_initializer='glorot_normal', bias_initializer='zeros'),
                keras.layers.Dense(width, activation=tf.nn.relu, kernel_initializer='glorot_normal', bias_initializer='zeros'),
                keras.layers.Dense(width, activation=tf.nn.relu, kernel_initializer='glorot_normal', bias_initializer='zeros'),
                keras.layers.Dense(width, activation=tf.nn.relu, kernel_initializer='glorot_normal', bias_initializer='zeros'),
                keras.layers.Dense(width, activation=tf.nn.relu, kernel_initializer='glorot_normal', bias_initializer='zeros'),
                keras.layers.Dense(width, activation=tf.nn.relu, kernel_initializer='glorot_normal', bias_initializer='zeros'),
                keras.layers.Dense(width, activation=tf.nn.relu, kernel_initializer='glorot_normal', bias_initializer='zeros'),
                keras.layers.Dense(width, activation=tf.nn.relu, kernel_initializer='glorot_normal', bias_initializer='zeros'),
                keras.layers.Dense(numOutput, activation=tf.nn.softmax)])
    

    unlabeledData=np.array(dataUnlabeled)
    labelForData=np.array(dataLabels)
    network.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    network.fit(unlabeledData,labelForData,epochs)
    return network






def test_set(data, network):
    dataUnlabeled=[np.array(x[0:(len(data[0])-1)], ndmin=2) for x in data]
    dataLabels=np.array([x[-1] for x in data])
    unlabeledData=np.array(dataUnlabeled)
    labelForData=np.array(dataLabels)
    (loss, accuracy)=network.evaluate(unlabeledData, labelForData)
    return (loss, accuracy)


#adds bias weight before the label
def fixData(data):
    for x in data:
        label=float(x[-1])
        x[-1]=1
        if label==0:
            x.append(0)
        if label==1:
            x.append(1.0)
        for i in range(len(x)-1):
            x[i]=float(x[i])
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
            (loss,accuracy)=test_set(testIdx,network)
            print('The loss on the test data is: '+str(loss)+' and the accuracy is: '+str(accuracy))
            (loss, accuracy)=test_set(idx,network)
            print('The loss on the training data is: '+str(loss)+' and the accuracy is: '+str(accuracy))
