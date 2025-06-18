# MachineLearningSandbox

This is a machine learning library developed by Theresa Sheets at the University of Utah.


To run id3.py
To run this code you need at least the args for test, train, and maxDepth change the files for test and train, and the depth for max depth as you choose.
You also need to include —-car or —-bank to specify either the cars or the bank dataset.

python id3.py --test test.csv —train train.csv -—maxDepth 6 —-cars

Optional args include -—unknowns to replace missing data, --me will specify majority error, and --gini will specify the gini index, otherwise the code will default to entropy.  



To run adaboost.py:
	python adaboost.py --train bank/train.csv --test bank/test.csv --bank --maxDepth 1 --T 5
	
	To change the number of iterations vary --T

To run baggedTrees.py:

	python baggedTrees.py --train bank/train.csv --test bank/test.csv --maxDepth 16 --T 5

	To change the number of iterations vary --T


To run randomForest.py:

	python randomForest.py --train bank/train.csv --test bank/test.csv --maxDepth 16 --T 5 --features 2

	To change the number of iterations vary --T
	To change the number of features vary --features



To run gradientDescent.py:

	python gradientDescent.py --train concrete/train.csv --test concrete/test.csv --r .01

	To change learning rate vary --r
	To test stochastic gradient descent add the arg --stochastic
