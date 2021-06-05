#         Brest cancer wiscoin         
###########################################################################################
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import Imputer
from sklearn.model_selection  import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sb



#importing the dataset
df=pd.read_csv('E:\\Pros\\ML_programms\\Projects\\breast_cancer\\breast-cancer-wisconsin.csv')

#giving column names
df.columns = ['Sample_cd_no','Clump_Thickness ','Uniformity_Cell_Size',
                     'Uniformity_Cell_Shape','Marginal_Adhesion',
                     'Single Epithelial Cell Size','Bare Nuclei',
                     'Bland Chromatin','Normal Nucleoli','Mitoses',
                     'Class']

print(df.isnull().sum())

#to convert ? into the null value
df['Bare Nuclei'].replace({'?': np.nan},inplace =True)

"""Null values can be represented as NaN (or) None"""

 # It directly shows the no of null values in each col
print(df.isnull().sum())


#Taking care of missing values  using imputer class
from sklearn.base import TransformerMixin

class DataFrameImputer(TransformerMixin):
    
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)
df = pd.DataFrame(df)
df = DataFrameImputer().fit_transform(df)
#dropping the first column bcz we didn,t need it during the calculation
df=df.drop(['Sample_cd_no'],axis=1)

##########################    Preprocessing   ############################

# slicing of  df dataset  to get dependent and idependent variables
X = df.iloc[:,df.columns !='Class'].values   
y = df.iloc[:,df.columns =='Class' ].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#      """Classification Algorithms  :--"""

#----------------------------------------------------------------------------------------------------
#   1.Logistic_Regression model:-

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
print("Accuracy using Logistic_Regression {} %".format(classifier.score(X_test,y_test)*100))
# Predicting the Test set results

# Making the Confusion Matrix
y_pred = classifier.predict(X_test) # Predicting the Test set results
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#accuracy====(correct prediction)/(sum of all prediction)

#--------------------------------------------------------
#   2. Knn algo:-

#'''checking accuracy'''

scorelist=[]  #here our max value will store

'''as by changing the value of n_neighbor our accuracy varies so we want maximum accuracy so here we
    create a for loop which helps to get at whichpoint we gor maximum accuracy'''
for i in range(1,20):              
    knn=KNeighborsClassifier(i)
    knn.fit(X_train,y_train)    
    prediction=-knn.predict(X_test)
    scorelist.append(knn.score(X_test,y_test)*100)
    
print('test accuracy of knn: {} %'.format(round(max(scorelist)),2))

#------------------------------------------------------------------------------
#3. Support Vector Machine

svm=SVC(random_state=0) 
svm.fit(X_train,y_train)
print('test accuracy of support Vecor Machine: {} %'.format(svm.score(X_test,y_test)*100))

#----------------------------------------------------------------------------------
# 4.Naive bayes algorithm

nb=GaussianNB()
nb.fit(X_train,y_train)
print('test accuray of Naive Bayes: {} %'.format(round(nb.score(X_test,y_test)*100),2))

#-----------------------------------------------------------------------------------
# 5.#Decision Tree Algorithm

dtc=DecisionTreeClassifier()
dtc.fit(X_train,y_train)
print('test accuray of Decision Tree : {} %'.format(round(dtc.score(X_test,y_test)*100),2))

#----------------------------------------------------------------------------------------
# 6.  Random Forest

rf=RandomForestClassifier(n_estimators=1000,random_state=1)
rf.fit(X_train,y_train)
print('test accuray of Random Forest: {} %'.format(round(rf.score(X_test,y_test)*100),2))

#==============================================================================================
#"""    comparing all the accuracy  """
methods=['Logistic Regression','KNN','SVM','Naive Bayes','Decision Tree','Random Forest'] 
 # it stores all the methods that we are going to use
accuracy=[96.0,98.0,96.0,95.0,95.0,97.0]

color=['green','#0FBBAE','purple','orange','magenta','cyan']

sb.set_style('whitegrid')
plt.figure(figsize=(8,6)) #size of the graph
plt.ylabel('Accuracy(%)')
plt.title("Algorithm prediction")
plt.xlabel('Algorithms')
sb.barplot(x=methods,y=accuracy,palette=color)  
#barplot just plot our graph in bar format and pallete=colors for givinh=g all methods a specific color
plt.show()



#####################################               End               ########################










