import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection  import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#import the data from the csv file from the path given
data=pd.read_csv("C:/Users/Sahil/Desktop/heart.csv")

'''gathering of data i.e. choosing of desired input & output'''
y=data.target.values    #extracting the target values from the data(or csv file) from range 0 to 1

#deleting target column from the data(or csv file) and stores the rest value in x_data

x_data=data.drop(['target'],axis=1)   #i/p values

#normalisation of i/p data in the range 0 to 1(bcz as our target values in the range 0 to 1 )
x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data)).values


''' prepare the data(spliting the data i.e testing and traing data '''

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
'''(as i want to get 20% of test data and 80% of train data by test_size=0.2
and randam_state=0 so no rabdom value will be choosen)'''

#choosing a model::

# logistic regression algorithm
log=LogisticRegression()  
log.fit(x_train,y_train)  

'''checking the accuracy'''
print('test accuracy of logistic regression: {}'.format(log.score(x_test,y_test)*100))

#knn algorithm

'''checking accuracy'''

scorelist=[]  #here our max value will store
for i in range(1,20):               #as by changing the value of n_neighbor our accuracy varies so we want maximum accuracy so here we createa for loop which helps to get at whichpoint we gor maximum accuracy
    knn=KNeighborsClassifier(i)
    knn.fit(x_train,y_train)    
    prediction=-knn.predict(x_test)
    scorelist.append(knn.score(x_test,y_test)*100)
    
print('test accuracy of knn: {}'.format(max(scorelist)))
'''
plt.plot(range(1,20),scorelist) #ploting a graph between 1 to 20 vs the max value
plt.xlabel('k value')
plt.ylabel('score')
plt.show()'''
