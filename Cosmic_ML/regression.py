import pandas as pd
#importing pandas to load the data from the excel file(.csv format)
import matplotlib.pyplot as plt
#importing matplot library for scattering of data
import numpy as np
#the values of housing dataset are a series but we want it in array format.so,numpy is used
from sklearn import linear_model
#generating a regression model


#gathering data
data=pd.read_csv('C:/Users/Sahil/Desktop/housing.csv')

y=np.array(data['HOUSE'])
x=np.array(data['IR'])
#as i want another shape of data so, reshape the variable

x=x.reshape(len(x),1)
y=y.reshape(len(y),1)

#preparing/splliting the data into training ans test data
x_train=x[:-250]  #training data values starts from 0 indexing and ending upto -251 indexing
                 

x_test=x[-250:]  #test data values starts from -250 indexing and ending at last element
                 
y_train=y[:-250]
y_test=y[-250:]

#chhosing a regression model
regr=linear_model.LinearRegression()

regr.fit(x_train,y_train)
             
#scattering of  test data (plotting the graph)
plt.scatter(x_test,y_test,color='green  ')
plt.plot(x_test,regr.predict(x_test),color='red',linewidth=3)             
plt.title('test data')
plt.ylabel('size')
plt.xlabel('price')
plt.show()
