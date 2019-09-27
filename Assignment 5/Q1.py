# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 21:45:38 2019

@author: Deepika
"""
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


# Import training and test data
def importTrainingdata():
    Training_data = pd.read_csv(r'C:\Users\Deepika\Downloads\studies\data mining\homework 5\PB1_train.csv',
                                names=['Height', 'Age', 'Weight', 'Class'])
    # print(Training_data)

    return Training_data


def importTestdata():
    Test_data = pd.read_csv(r'C:\Users\Deepika\Downloads\studies\data mining\homework 5\PB1_test.csv',
                            names=['Height', 'Age', 'Weight', 'Class'])
    #    print(Test_data)
    #
    return Test_data


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


def train_using_NaiveBayes(X_train, Y_train):
    # Creating the classifier object
    gnb = GaussianNB()
    gnb.fit(X_train, Y_train)
    return gnb


def prediction(X_test, gnb):
    # Predicton on test with giniIndex
    y_pred = gnb.predict(X_test)
    #   print("Predicted values:")
    print("y_pred:", y_pred)
    return y_pred


def main():
    # Building Phase
    Trainingdata = importTrainingdata()
    Testdata = importTestdata()
    X_train, X_test, Y_train, Y_test = SplitData(Trainingdata, Testdata)
    gnb = train_using_NaiveBayes(X_train, Y_train)

    # Prediction using Naive Baye's
    y_pred_NaiveBayes = prediction(X_test, gnb)
    print(metrics.accuracy_score(Y_test, y_pred_NaiveBayes) * 100)


## Calling main function
if __name__ == "__main__":
    main()
