# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv("Data.csv")
#distinguishing bn independent and dependent coloumns in dataset
X =  dataset.iloc [:,  :-1].values
Y=dataset.iloc[:,3]
#importing Imputer class from sklearn.preprocessing library to find the mean of the coloumns
from sklearn.preprocessing import Imputer
#creating object imputer for Imputer class
imputer=Imputer(missing_values='NaN', strategy='mean',axis=0)
#to fit the imputer object into the matrix
imputer=imputer.fit(X[:,1:3])
#to replace the missing value in the table
X[:,1:3]=imputer.transform(X[:,1:3])
#categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#creating objects for both the classes
labelencoder_X=LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])
onehotencoder=OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()
labelencoder_y=LabelEncoder()
Y=labelencoder_y.fit_transform(Y)
#splitting data into test and train sets
from sklearn.cross_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)