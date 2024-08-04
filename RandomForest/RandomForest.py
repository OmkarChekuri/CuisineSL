import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import csv
import pandas as pd
import numpy as np
import pickle
import time
import math
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier 

print('Loading train data...')
# Reading the yummly dataset from the location
with open('./train.json', encoding='utf-8', errors='replace') as f:
    data = f.read()[3:-3]
    #print(data)
    data = data.split("},")
    data.append("dummy")
    meals = []
    for each in data[:-1]:
        each = each + "}"
        meals.append(json.loads(each))

#list for ingredients , id , cuising
itemList = []
itemID = []
meal_cuisine = []

# split the json file into id, cuisine and ingredients respectively
for each in meals:
    m = ""
    itemID.append(each['id'])
    meal_cuisine.append(each['cuisine'])
    for each1 in each['ingredients']:
        #replace space in the ingredients with underscore
        each1 = each1.replace(' ', '_')
        m += each1 + ' '
    itemList.append(m)

#convert the ingredients into array of 0 ,1 for train data
vectorizer = CountVectorizer()
vectors = vectorizer.fit_transform(itemList).toarray() 
vectors = pd.DataFrame(data=vectors)
meal_cuisine = pd.DataFrame([meal_cuisine])


#test data
print('\nLoading test data...')
with open('./test.json', encoding='utf-8', errors='replace') as g:
    data = g.read()[3:-3]
    #print((data))
    data = data.split("},")
    data.append("dummy")
    meals_test = []
    for each in data[:-1]:
        each = each + "}"
        meals_test.append(json.loads(each))
    
#list for ingredients, id, cuisine
itemList_test = []
itemID_test = []
meal_cuisine_test = []

# split the json file into id, cuisine and ingredients respectively
for each in meals_test:
    m = ""
    itemID_test.append(each['id'])
    #meal_cuisine_test.append(each['cuisine'])
    for each1 in each['ingredients']:
        #replace space in the ingredients with underscore
        each1 = each1.replace(' ', '_')
        m += each1 + ' '
    itemList_test.append(m)

#convert all of the ingredients into array of unique value (0 or 1) for test data, using the train vocabulary
vectorizer2 = CountVectorizer(vocabulary=vectorizer.vocabulary_)
vectors_test = vectorizer2.fit_transform(itemList_test).toarray()

features_train = vectors

target_train = meal_cuisine.T

#target_train = target_train.rename(columns = {0: "Cuisine"}, inplace = True) 

training_data = pd.concat([features_train, target_train], axis=1, sort=False)

#print(features_train.shape)
#print(target_train.shape)
#print(training_data.head())
#training_data = training_data.rename(columns={0: 'Cuisine'})
#print(training_data.head(2))

columnNames = vectorizer.get_feature_names()

columnNames.append("cuisine")

training_data.columns = columnNames

print(training_data.head())


Train = training_data

Train_X = Train.iloc[:,0:6867]
Train_Y = Train.iloc[:,-1:]
Test = pd.DataFrame(vectors_test)
start_time = time.time()

RandomForestModel = RandomForestClassifier(n_estimators = 180, criterion = 'gini', max_features = "sqrt" , max_depth = None, bootstrap = True) 


RandomForestModel.fit(Train_X,Train_Y)


print("--- %s seconds ---" % (time.time() - start_time))
#tree.plot_tree(decision_tree_gini.fit(Train_X,Train_Y))
predicted_cuisineRF300 = RandomForestModel.predict(Test)
#print(predicted_cuisine)
data = {"id":itemID_test,"cuisine":predicted_cuisineRF300}
dataframe = pd.DataFrame(data)
dataframe.to_csv('Submission_RF_maxDepth_NoneAllDataMaxFeatures_sqrt_nestimators200.csv', index=False)
print('writing Predictions to csv file complete')