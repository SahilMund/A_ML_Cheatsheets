# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#importing the data set
dataset = pd.read_csv("Data.csv")
#distinguishing bn independent and dependent coloumns in dataset
X =  dataset.iloc [:,  :-1].values
Y=dataset.iloc[:,3]

#splitting data into test and train sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
#Feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)"""