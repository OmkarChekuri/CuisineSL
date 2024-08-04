import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import tree
import csv
import pandas as pd
import numpy as np
import pickle
import time
import math
from questionClass import Question
from nodeClass import Node
from decisionTreeClass import DecisionTree


def main():
    
    #Data Preparation
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

    columnNames = vectorizer.get_feature_names()

    columnNames.append("cuisine")

    header = columnNames

    training_data.columns = columnNames

    print(training_data.head())

    trainingdata2 = training_data.values.tolist()
    #print(trainingdata2[0:5])

    Train = training_data
    Train_X = Train.iloc[:,0:6867]
    Train_Y = Train.iloc[:,-1:]
    '''
    with open("trainingDataPkl.txt", "wb") as fp:
        #Pickling
        pickle.dump(trainingdata2, fp)

    with open("trainingDataPkl.txt", "rb") as fp:
        # Unpickling
        trainingdata2 = pickle.load(fp)

    with open("columnNamesPkl.txt", "wb") as fp:
        #Pickling
        pickle.dump(columnNames, fp)
    
    with open("columnNamesPkl.txt", "rb") as fp:
        # Unpickling
        columnNames = pickle.load(fp)
    print(columnNames[1:10])

    '''



    start_time = time.time()
    #create the model
    DT_prune = DecisionTree( min_samples_split=5, max_depth=500)
    #fit the model
    DT_prune.fit(Train_X, Train_Y,DT_prune.getnode())
    print("--- %s seconds ---" % (time.time() - start_time))
    
    #Make predictions
    predicted_cuisine = []
    for example in vectors_test:
   
        predicted_cuisine.append(DT_prune.predict(example,DT_prune.getnode() ))

    data = {"id":itemID_test,"cuisine":predicted_cuisine}
    dataframe = pd.DataFrame(data)
    dataframe.to_csv('ScratchSubmission_DT_gini_Pruning_1000rows.csv', index=False)
    print('writing Predictions to csv file complete')


if __name__ == 'main':
    main()