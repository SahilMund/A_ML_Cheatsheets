# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#Importing the dataset. Here we import tsv file
dataset=pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3, engine='python')#engine=python is added just to avoid the warning

#cleaning the text
import re
import nltk
nltk.download('stopwords')#to download a special tool called stopwords. It is a list od all the irrelevant words
from nltk.corpus import stopwords#stopwords id the list containing all the irrelevant words of each language
from nltk.stem.porter import PorterStemmer#importing class for stemming
ps=PorterStemmer()#obj for porter stemmer class
corpus=[]
 #it will store 1000 cleaned reviews
for i in range(0,1000):
    #First we will clean the !st review
    review=re.sub('[^a-zA-z]',' ',dataset['Review'][i])
    review=review.lower() #converting all the alphabets in the review to lower case
        
    #Predicting the non relevant words in the review.
    review=review.split()#splits different words of a string into elements of a list
    review=[ps.stem(word) for word in review if not word in stopwords.words('english')]#keeps the relevant words in the list
    
    #joining different words in the review srting and seprating them with a space
    review=' '.join(review)#here we get the cleaned review
    corpus.append(review)#Here we add all the reviews to the corpus list
    
#Creating the Bag of Words Model
#creating the Sparse matrix
from sklearn.feature_extraction.text import CountVectorizer
Cv=CountVectorizer(max_features=1500)
X=Cv.fit_transform(corpus).toarray()

#Training our ML mo del
#splitting it into dependent and independent variables
y=dataset.iloc[:,1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting classifier to the Training set
# Create your classifier here
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) 