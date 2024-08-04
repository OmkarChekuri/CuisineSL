#node class to represent the node of the decision tree
class Node:
    def __init_(self,left = None, right = None, feature = None,*, classlabel = None):
        self.feature = question
        self.left = true_branch
        self.right = false_branch
        self.classlabel = classlabel
        
        
    def setclasslabel(self,predictions):
        classLabelsdict = {} 
        
        for prediction  in predictions:
            if not(prediction in classLabelsdict.keys()):
                classLabelsdict[prediction] = 0
            else:
                classLabelsdict[prediction] += 1
        self.classlabel = max(classLabelsdict, key=classLabelsdict.get) 
        
    def getclasslabel(self):
        return self.classlabel
    
    
    def isleaf(self):
        isleafnode = False
        if self.classlabel == None:
            isleafnode = True
        else:
            isleafnode = False
        
        return isleafnode
        
