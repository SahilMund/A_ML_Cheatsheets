# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#creating objects for both the classes
labelencoder_X_1=LabelEncoder()
X[:,1]=labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2=LabelEncoder()
X[:,2]=labelencoder_X_2.fit_transform(X[:,2])
onehotencoder=OneHotEncoder(categorical_features=[1])
X=onehotencoder.fit_transform(X).toarray()
#removing dummy variable trap
X=X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Part 2. Creating our ANN model
#importing the keras libraries and packages
import keras
from keras.models import Sequential #Req to initiate the Neural Network
from keras.layers import Dense#req to create layers in our ANN

#initializing the ANN
classifier=Sequential()
#Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim=6, input_dim=11, init='uniform', activation='relu'))
#Adding the second hidden layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))
#Adding the Output layer
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

#compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Fitting the ANN to training set
classifier.fit(X_train,y_train,batch_size=10, nb_epoch=100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
#predicting a customer stays or leaves using the probabilities in Y_pred
y_pred=(y_pred>0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

