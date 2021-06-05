#Simple Linear Regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#importing the data set
dataset = pd.read_csv("Salary_Data.csv")
#distinguishing bn independent and dependent coloumns in dataset
X =  dataset.iloc [:,  :-1].values
Y=dataset.iloc[:, 1]
#splitting data into test and train sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)
#Feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)"""
#Fitting SIR to training set
#importing library
from sklearn.linear_model import LinearRegression
#creating object for LinearRegression class
regressor=LinearRegression()
regressor.fit(X_train,y_train)
#predicting the test set resuts
y_pred=regressor.predict(X_test)
#plotting the graphs for training set results
#Here observation of real values will be plotted the real values of X_train is compared to values predicted from x_train
plt.scatter(X_train,y_train,color = 'red')
#plotting the regerssion line
plt.plot(X_train,regressor.predict(X_train), color = 'blue')
#we will give a title to our graph and label the X and Y axis(just formalities)
plt.title('Salary vs Experience(Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
#plotting the graphs for test set results
plt.scatter(X_test,y_test,color = 'red')
#plotting the regerssion line
plt.plot(X_train,regressor.predict(X_train), color = 'blue')
#we will give a title to our graph and label the X and Y axis(just formalities)
plt.title('Salary vs Experience(Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()