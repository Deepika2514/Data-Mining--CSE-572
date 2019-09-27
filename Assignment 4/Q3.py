# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 10:41:39 2019

@author: Deepika
"""

# Run this program on your local python
# interpreter, provided you have installed
# the required libraries.

# Importing the required packages
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# Function importing Dataset
def importTrainingdata():
    Training_data = pd.read_csv(r'C:\Users\Deepika\Downloads\studies\data mining\homework4\PB3_train.csv',
                                names=['Height', 'Age', 'Weight', 'Class'])
    # print(Training_data)
    # Printing the dataswet shape

    return Training_data


def importTestdata():
    Test_data = pd.read_csv(r'C:\Users\Deepika\Downloads\studies\data mining\homework4\PB3_test.csv',
                            names=['Height', 'Age', 'Weight', 'Class'])
    #    print(Test_data)
    #    # Printing the dataswet shape
    #    print ("Dataset Lenght: ", len(Test_data))
    #    print ("Dataset Shape: ", Test_data.shape)
    #
    #    # Printing the dataset obseravtions
    #    print ("Dataset: ",Test_data.head())
    return Test_data


## Function to split the dataset
def SplitData(Training_data, Test_data):
    # Seperating the target variable
    X_train = Training_data.values[:, 0:3]
    #    print("X_train:",X_train)
    #    print(X_train.shape)
    Y_train = Training_data.values[:, 3]
    # print("Y_train:",Y_train)
    X_test = Test_data.values[:, 0:3]
    print("X_test:", X_test)
    Y_test = Test_data.values[:, 3]
    print("Y_test:", Y_test)

    return X_train, X_test, Y_train, Y_test


#
## Function to perform training with giniIndex.
def train_using_gini(X_train, X_test, Y_train):
    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion="gini")

    # Performing training
    clf_gini.fit(X_train, Y_train)
    return clf_gini


## Function to make predictions
def prediction(X_test, clf_object):
    # Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)
    #    print("Predicted values:")
    print("y_pred:", y_pred)
    return y_pred


## Function to calculate accuracy
def cal_accuracy(Y_test, y_pred):
    print("Confusion Matrix: ",
          confusion_matrix(Y_test, y_pred))

    print("Accuracy : ",
          accuracy_score(Y_test, y_pred) * 100)

    print("Report : ",
          classification_report(Y_test, y_pred))


## Driver code
def main():
    # Building Phase
    Trainingdata = importTrainingdata()
    Testdata = importTestdata()
    X_train, X_test, Y_train, Y_test = SplitData(Trainingdata, Testdata)
    clf_gini = train_using_gini(X_train, X_test, Y_train)

    # Prediction using gini
    y_pred_gini = prediction(X_test, clf_gini)
    cal_accuracy(Y_test, y_pred_gini)


## Calling main function
if __name__ == "__main__":
    main()