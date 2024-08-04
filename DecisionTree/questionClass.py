

#class file to define the question

class Question:

    def __init__(self, attributeIndex, value):
        self.attributeIndex = attributeIndex
        self.value = value
    #check if the example matches the value of the question
    def isExist(self, example):
        self.value = example[self.attributeIndex]
        return val == self.value
 