# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
#Here we will not split it into training and test sets as we have very less data. So we will use all the data to train our machine
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""# -*- coding: utf-8 -*-
#fitting linear regression to the model
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X, y)
#fitting Polynomial regresion to data set
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
#creating the matrix which includes the polinomial terms in the model
#first fit the poly_reg obj to X then transform X to X_poly
X_poly=poly_reg.fit_transform(X)
poly_reg.fit(X_poly,y)
#We will include the fit to the a PLR models
lin_reg_2=LinearRegression()
lin_reg_2.fit(X_poly,y)
#Plotting LR results(here we will plot the true observation points)
plt.scatter(X,y,color='red')
#Now wew will plot the predicted values of X according to LR
plt.plot(X,lin_reg.predict(X),color='blue')
#formatting the graph
plt.title('Truthor Bluff(Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
#Plotting Polynomial Regression results(here we will plot the true observation points)
plt.scatter(X,y,color='red')
#Now wew will plot the predicted values of X according to Polynomial Regression
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)),color='blue')
#formatting the graph
plt.title('Truthor Bluff(Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
#plotting a graph for a difference of 0.1 in levels in place of 1
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid,lin_reg_2.predict(poly_reg.fit_transform(X_grid)),color='blue')
plt.title('Truthor Bluff(Polynomial Regression) 0.1 level difference')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
#predicting the salary of the employee at level 6.5 using LR model. But it is not relevant according to the plot
lin_reg.predict([[6.5]])
#wee can also use thethe following code for the above line[lin_reg.predict(np.array(6.5).reshape(1,-1))]
#Predicting the salary of level 6.5 according to PLR.
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
#we can also use this code in the above line [lin_reg_2.predict(poly_reg.fit_transform(np.array(6.5).reshape(1,-1)))]