#Theresa Sheets ID3 Algorithm Implimentation
#May 4th
#CMSC 5350/6350
#For Final Project


import argparse, sys, math


parser = argparse.ArgumentParser()
parser.add_argument( "--train", help="Trains the set." )
parser.add_argument( "--test", help="Tests the set." )
parser.add_argument( "--car", help="Use the Car attributes", action="store_true" )
parser.add_argument( "--bank", help="Use the Bank attributes",action="store_true" )
parser.add_argument("--CF",help="Use the CF attributes", action="store_true")
parser.add_argument( "--me", help="Use majority_error", action="store_true" )
parser.add_argument( "--gini", help="Use gini index", action="store_true" )
parser.add_argument( "--maxDepth", help="What is the max depth", type=int )
parser.add_argument( "--unknowns", help="Fix unknowns in the set", action="store_true" )

args = parser.parse_args()

#median function
def median(lst):
    n = len(lst)
    if n < 1:
            return None
    if n % 2 == 1:
        return sorted(lst)[n//2]
    else:
        return sum(sorted(lst)[n//2-1:n//2+1])/2.0


#makes nodes
class Node(object):
    def __init__(self, data,label, entropy, maxDepth):
        self.data=data
        self.value = "base" #the value of the atribute which caused the branch
        self.key_attr= "base" #how previous node was split
        self.label=label
        self.children = []
        self.node_entropy = entropy
        self.node_maxDepth=maxDepth


    def print_node(self):
        print("This node is at depth:"+str(self.node_maxDepth))
        print("This node was created by branching on the attribute "+(self.key_attr)+" and has value "+str(self.value)+" of that attribute")
        print("There are "+str(len(self.data))+" training data entries at this node.")
        print("The label of this node is "+ str(self.label))
        print("This node has entropy: "+str(self.node_entropy))
        print("This node has: "+ str(len(self.children))+ " children\n")
        for x in self.children:
            x.print_tree()


#Decision tree class
class ID3Tree(object):
    def __init__(self, data, attributes, labels, maxDepth,value,key_attribute):
        self.data = data
        self.attributes = attributes # a list of x values in the column should just be numbers
        self.labels = labels #(unacceptable, acceptable, good, verygood)
        self.root = None
        #self.depth=0
        self.max_depth = maxDepth
        self.value=value
        self.node_attribute=key_attribute
        self.at_root = False


    #picks the attribute which is best to split the set
    def best_split_set(self):
        best_attribute = [0 for attr in range(0,len(self.attributes))]
        idx=0
        for attribute in self.attributes:
            best_attribute[idx]=self.gain(attribute)
            idx+=1
        best_split = max(best_attribute)
        idx=0
        for gain in best_attribute:
            if gain==best_split:
                break
            idx+=1
        best_index = idx
        return self.attributes[best_index]


    #gives attribute_data
    def get_attribute_values(self, attribute):
        if args.car:
            values ={
                'buying':['vhigh','high','med', 'low'],
                'maint':['vhigh','high','med', 'low'],
                'doors':['2','3','4','5more'],
                'persons':['2','4','more'],
                'lug_boot':['small','med','big'],
                'safety':['low','med','high']
            }
        if args.bank and self.at_root:
            values ={
                'age':[],
                'job':['admin.','unknown','unemployed','management', 'housemaid','entreprenuer','student','blue-collar','self-employeed','retired','technician','services'],
                'marital':['married', 'divorced', 'single'],
                'education':['unknown','secondary','primary','tertiary'],
                'default':['yes','no'],
                'balance':[],
                'housing':['yes','no'],
                'loan':['yes','no'],
                'contact':['unknown','telephone','cellular'],
                'day':[],
                'month':['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'],
                'duration':[],
                'campaign':[],
                'pdays':[],
                'previous':[],
                'poutcome':['unknown','other','failure','success']
            }
        if args.bank and not self.at_root:
            values={
                'age':['low','high'],
                'job':['admin.','unknown','unemployed','management', 'housemaid','entreprenuer','student','blue-collar','self-employeed','retired','technician','services'],
                'marital':['married', 'divorced', 'single'],
                'education':['unknown','secondary','primary','tertiary'],
                'default':['yes','no'],
                'balance':['low', 'high'],
                'housing':['yes','no'],
                'loan':['yes','no'],
                'contact':['unknown','telephone','cellular'],
                'day':['low', 'high'],
                'month':['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'],
                'duration':['low', 'high'],
                'campaign':['low', 'high'],
                'pdays':['low', 'high'],
                'previous':['low', 'high'],
                'poutcome':['unknown','other','failure','success']
            }
        if args.CF and self.at_root:
            values={
                'mssa':['0','1'],
                'mrsa':['0','1'],
                'h_flu':['0','1'],
                'pseudo':['0','1'],
                'burkho_complex':['0','1'],
                'alcalig':['0','1'],
                'steno':['0','1'],
                'enterobacter':['0','1'],
                'serratia_marcescens':['0','1'],
                'aspergillus':['0','1'],
                'candida':['0','1'],
                'scedosporium':['0','1'],
                'mabscessus':['0','1'],
                'mai':['0','1'],
                'sex':['0','1'],
                'suff':['0','1'],
                'diabet':['0','1'],
                'impglu':['0','1'],
                'fev1pct_best':[],
                'zscore_best':[]
            }
        if args.CF and not self.at_root:
            values={
                'mssa':['0','1'],
                'mrsa':['0','1'],
                'h_flu':['0','1'],
                'pseudo':['0','1'],
                'burkho_complex':['0','1'],
                'alcalig':['0','1'],
                'steno':['0','1'],
                'enterobacter':['0','1'],
                'serratia_marcescens':['0','1'],
                'aspergillus':['0','1'],
                'candida':['0','1'],
                'scedosporium':['0','1'],
                'mabscessus':['0','1'],
                'mai':['0','1'],
                'sex':['0','1'],
                'suff':['0','1'],
                'diabet':['0','1'],
                'impglu':['0','1'],
                'fev1pct_best':['low','high'],
                'zscore_best':['low','high']
            }
        return values.get(attribute,[])


    #calculates the gain for a particular attribute
    def gain(self, attribute):
        #changed previous line, self.entropy(self.labels) to current
        set_entropy = self.entropy(self.data)
        attribute_number = self.attributes.index(attribute)
        attribute_values=self.get_attribute_values(attribute)
        attribute_scores=[0 for i in range(0,len(attribute_values))]
        gain=set_entropy
        for x in attribute_values:
            subset=[]
            for row in self.data:
                if row[attribute_number] == x:
                    attribute_scores[attribute_values.index(x)]+= 1
                    subset.append(row)
            subset_entropy=self.entropy(subset)
            gain=gain-attribute_scores[attribute_values.index(x)]*subset_entropy/len(self.data)
        return gain


    #calculates the entropy of a set
    def entropy(self, myset):
        dataLength=len(myset)
        if args.me:
            majority_error=0
            if dataLength==0:
                return majority_error
            majority=self.max_label(myset)
            majority_count=self.number_labels(majority, myset)
            majority_error=1-(majority_count / float(dataLength))
            return majority_error
        if args.gini:
            gini=1
            if dataLength==0:
                return gini
            for x in self.labels:
                p=self.number_labels(x,myset) / float(dataLength)
                gini-=p*p
            return gini
        entropy=0
        if dataLength==0:
            return entropy
        for x in self.labels:
            p=self.number_labels(x,myset) / float(dataLength)
            if p!=0:
                entropy-=p*math.log(p,2)
        return entropy


    #returns the label which occurs most often
    def max_label(self,myset):
        count=0
        majorLabel=self.labels[0]
        for x in self.labels:
            numberForLabel=self.number_labels(x,myset)
            if numberForLabel> count:
                count=numberForLabel
                majorLabel=x
        return majorLabel


    #tells you how many times a specific response appears
    def number_labels(self, label, myset):
        num_label=0
        for x in myset:
            if x[-1] ==label:
                num_label = num_label+1
        return num_label


    #tells you if all the labels match and which label is the one they all match
    def all_match_labels(self):
        '''
        {
            "unacceptable": 30,
            "acceptable":
            "good": 20,
            "very good": 500
        }
        '''
        match_label=self.data[0][-1]
        all_match=True
        for x in self.data:
            if x[-1]!=match_label:
                all_match=False
                break
        return (all_match, match_label)


    #prints tree
    def print_tree(self):
        if (self.root):
            x=self.root
            x.print_node()


    def most_common_attr_value(self,attr):
        attr_index=self.attributes.index(attr)
        attribute_values=self.get_attribute_values(attr)
        attr_count=[0 for x in range(0,len(attribute_values))]
        for x in self.data:
            idx=0
            for attr_val in attribute_values:
                if x[attr_index]==attr_val:
                    attr_count[idx]+=1
                idx+=1
        max_val=max(attr_count)
        max_index=attr_count.index(max_val)
        if attribute_values[max_index]=='unknown':
            attr_count.remove(max_val)
            max_val=max(attr_count)
            max_index=attr_count.index(max_val)
        return attribute_values[max_index]


#actually does algorithm
    def driver_script(self):
        medians=[]
        mostCommonValues=[]
        if self.max_depth==args.maxDepth:
            self.at_root=True
        if self.at_root:
            for attr in self.attributes:
                attribute_values=self.get_attribute_values(attr)
                attr_index=self.attributes.index(attr)
                if len(attribute_values)==0:
                    medianVector=[]
                    for x in self.data:
                        medianVector.append(float(x[attr_index]))
                    medianValue=median(medianVector)
                    medians.append(medianValue)
                    for x in self.data:
                        if float(x[attr_index])>=medianValue:
                            x[attr_index]='high'
                        else:
                            x[attr_index]='low'
                else:
                    #replace all unknown with most common value
                    if args.unknowns:
                        mostCommonValue=self.most_common_attr_value(attr)
                        mostCommonValues.append(mostCommonValue)
                        for x in self.data:
                            if x[attr_index]=='unknown':
                                x[attr_index]=mostCommonValue

            self.at_root=False

        key_attribute=self.best_split_set()
        (all_match, label)=self.all_match_labels()
        entropy=self.entropy(self.data)
        label=self.max_label(self.data)
        root_node= Node(self.data, label, entropy,self.max_depth)
        if self.max_depth < 0:
            #no underground nodes
            return
        else:
            if ((not all_match) and (self.max_depth>0)):
                attribute_values = self.get_attribute_values(key_attribute)
                #if not attribute_values:

                for v in attribute_values:
                    #1. add a new tree branch corresponding to A=v
                    subset_v=[]
                    for data in self.data:
                        if data[self.attributes.index(key_attribute)]==v:
                            subset_v.append(data)
                    if len(subset_v)!=0:
                        subtree=ID3Tree(subset_v,self.attributes, self.labels, self.max_depth-1,v,key_attribute)
                        root_node.children.append(subtree)
                        subtree.driver_script()
                        #2. set subset_v a subset of examples in A with A=v
                        #3a else:
                            #add subtree ID3(subset_v, attribute,label)
            self.root=root_node
            self.root.value=self.value
            self.root.key_attr=self.node_attribute
            #import pdb; pdb.set_trace()
            return (root_node, medians, mostCommonValues)



    def predict_label(self, x):
        label=self.root.label
        for child in self.root.children:
            #find attribute and attribute value for that node
            key_attribute=child.root.key_attr
            idx=0
            for attribute in self.attributes:
                if key_attribute==attribute:
                    attribute_index=idx
                idx+=1
            attribute_value=child.root.value

            #if attribute value matches call predict on that node
            if x[attribute_index]==attribute_value:
                label=child.predict_label(x)
        return label


def train_set( data ):
    if args.car:
        attributes=['buying', 'maint', 'doors','persons','lug_boot','safety']
        labels=['unacc','acc','good', 'vgood']
    if args.bank:
        attributes=['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome']
        labels=['yes','no']
    if args.CF:
        attributes=['mssa','mrsa','h_flu','pseudo','burkho_complex','alcalig','steno','enterobacter','serratia_marcescens','aspergillus', 'candida','scedosporium','mabscessus','mai','sex','suff','diabet','impglu','fev1pct_best','zscore_best']
        labels=['0','1']
    tree=ID3Tree(data,attributes,labels, args.maxDepth, "all","base")
    (rootOfTree, medians, mostCommonValues)=tree.driver_script()
    tree.print_tree()
    return (tree, medians, mostCommonValues)


#fixes the numerical values in the test set
def fixNumerical(data, numericalAttributes, medianValues):
    for i in range(len(numericalAttributes)):
        for x in data:
            if x[numericalAttributes[i]]>=medianValues[i]:
                x[numericalAttributes[i]]='high'
            else:
                x[numericalAttributes[i]]='low'


def test_set( data, tree, medians, mostCommonValues):
    correct_predictions=0
    length=len(data)
    if args.bank:
        dataSet='bank'
        numericalAttributes=[0, 5, 9, 11, 12, 13, 14]
        fixNumerical(data,numericalAttributes,medians)
    if args.CF:
        dataSet='CF'
        numericalAttributes=[18,19]
        fixNumerical(data,numericalAttributes,medians)
    if args.unknowns:
        for x in data:
            for i in range(len(mostCommonValues)):
                if x[i]=='unknown':
                    x[i]=mostCommonValues[i]
    for x in data:
        label=tree.predict_label(x)
        if label==x[-1]:
            correct_predictions+=1
    prediction_accuracy=correct_predictions / float(length)
    if args.car:
        dataSet='cars'
    print("Prediction Accuracy: "+str(prediction_accuracy)+" on the "+dataSet+" at depth: "+str(args.maxDepth))


def load_file( datFile):
    idx=[]
    with open( datFile, 'r' ) as inFile:
        for line in inFile:
            terms = line.strip().split(',')
            idx.append(terms)
    idx2=[]
    length=len(idx[1])
    if args.CF:
        for x in idx:
            idx2.append(x[1:length])
        idx=idx2
        #import pdb; pdb.set_trace()
    return idx

if __name__ == "__main__":
    if not args.train:
        print( "ERROR: data files were not supplied to train the tree" )
        sys.exit( -1 )
    else:
        idx=load_file( args.train )
        (tree, medians, mostCommonValues)=train_set(idx)
    if not args.test:
        print( "ERROR: data files were not supplied, there is nothing to train against.")
        sys.exit( -1 )
    else:
        idx=load_file( args.test )
        test_set(idx, tree, medians, mostCommonValues)

