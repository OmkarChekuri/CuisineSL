
from questionClass import Question
from nodeClass import Node

# Decision tree class
class DecisionTree:
    
    def __init__(self, min_samples_split=2, max_depth=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    #fit the model   
    def fit(self,x,y,node,depth = 0):

        #Calculate the split and feature at the root node
        information_gain, feature = self.BestSplitCriteria(x,y)
       
       # check stopping criteria and iteratively build the tree
        if (depth>=self.max_depth) and not(self.max_depth == None):
            leafnode == Node(value = None)
            leafnode.setclasslabel(y)
            self.root = leafnode
        elif information_gain == 0 :
            leafnode == Node(value = None)
            leafnode.setclasslabel(y)
            self.root = leafnode    
        else:
            self.root = self.buildtree(x,y, question)
        
        return self.root

    #bUild the tree
    def buildtree(self,x,y,feature):
        leftBranchRows, RightBranchRows = partition(x,y, feature)
        Left = self.fit(leftBranchRows.iloc[:,0:leftBranchRows.shape()[2]], leftBranchRows.iloc[:,-1:], node,depth +1)
        Right = self.fit(RightBranchRows.iloc[:,0:RightBranchRows.shape()[2]], RightBranchRows.iloc[:,-1:],node,depth +1)
        return Node(Left, Right,feature)


        
     #predict the classes   
    def predict(self, row, node):

        if isinstance(self.root, Leaf):
            return list(node.predictions.keys())[0]
        else:
            if self.root.question.isExist(row):
                return list(self.predict(row, node.left).keys())[0]
            else:
                return list(self.predict(row, node.right).keys())[0]


    def gini(self,trainY):
   
        counts = {}
        for idx,row in trainY.iterrows():
            label = row['cuisine']
            if label not in counts:
                counts[label] = 0
            counts[label] += 1    
            
        gini_value = 1
        for item in counts:
            probability = counts[item] / trainY.shape(2)
            gini_value -= probability**2
        return gini_value

    #calculate the feature with highest information gain        
    def BestCriteria(self,x,y):
        Best_gain = 0  
        Best_feature = None 
        current_uncertainty = self.gini(trainy)
        attributesCount = Train_X.shape[1]  

        for columnIndex in range(attributesCount):  

            uniqueValues = list(Train_X.iloc[:,columnIndex].unique())  

            for value in uniqueValues:  

                feature = Question(columnIndex, value)


                leftbranchSamples, rightBranchSamples = self.splitTheNode(trainx,trainy, feature)


                if leftbranchSamples.shape[1] == 0 or rightBranchSamples.shape[1] == 0:
                    continue

                prob = float(len(leftbranchSamples)) / (len(rightBranchSamples) + len(leftbranchSamples))
                informationgain = current_uncertainty - prob*gini(leftbranchSamples.iloc[:,-1:]) - (1 - prob)*gini(rightBranchSamples.iloc[:,-1:])

                if informationgain >= Best_gain:
                    Best_gain, Best_feature = gain, feature

        return Best_gain, Best_feature

    #split the node based on the feature
    def splitTheNode(self,train_X,train_Y, feature):

        traindata = pd.concat([train_X, train_Y], axis=1).values.tolist()
        leftbranchSamples = []
        rightBranchSamples = []
        for row in traindata:
            if feature.match(row):
                leftbranchSamples.append(row)
            else:
                rightBranchSamples.append(row)
        leftbranchSamples = pd.DataFrame(leftbranchSamples)
        rightBranchSamples = pd.DataFrame(rightBranchSamples)
        return leftbranchSamples, rightBranchSamples

        