# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 20:07:35 2019

@author: ASUS
"""
# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
#categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#creating objects for both the classes
labelencoder=LabelEncoder()
X[:, 3]=labelencoder.fit_transform(X[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()
#Avoiding Dummy variable Trap
X=X[:, 1:]
#no need of encoding dependent variable
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""
#fitting Multiple Linear Regression to training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
#predicting porfits for the test set of independent variables ie X_test
y_pred=regressor.predict(X_test)
#building the optimal model using backward elimination
import statsmodels.formula.api as sm
#here we are adding a coloumn of 1's to X. For details refer to notes
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
#including all the independent variables to a new matrix X_opt and later we will remove the cols which are not statistically significant
X_opt=X[:, [0,1,2,3,4,5]]
#now we will import a new class i.e OLS to fit MLR to X_opt and y
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
#we will use a function summary to know the p_values of the cols of matrix X_opt to remove the cols that are not statistically significant
regressor_OLS.summary()
#now ww will copy the above 3 lines and paste it and remove the cols if P_value>0.005 from X_opt untill the highest p_value is <=0.005
#here he highest p value is of x2 i.e index 2 which is far greater than 0.05. so we will remove it
X_opt=X[:, [0,1,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
#highest p_value is of x1 i.e index 1>than 0.05 so it is removed
X_opt=X[:, [0,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
#highest p_value is of x1 i.e index 4>than 0.05 so it is removed
X_opt=X[:, [0,3,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
#highest p_value is of x1 i.e index 5>than 0.05 so it is removed
X_opt=X[:, [0,3]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
#now thehighest p value is of x5 < 0.05. so here we will stop