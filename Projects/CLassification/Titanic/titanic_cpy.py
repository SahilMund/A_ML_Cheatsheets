import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

train_raw=pd.read_csv("train.csv")
test_raw = pd.read_csv("test.csv")
#searching the number of null values in each coloumn
print("In Training set")
null_counts_train = train_raw.isnull().sum()
null_counts_train[null_counts_train > 0].sort_values(ascending=False)
print("In test set")
null_counts_test = test_raw.isnull().sum()
null_counts_test[null_counts_test > 0].sort_values(ascending=False)
#dropping the cabin coloumn as it has high missing values
train=train_raw.drop(["Cabin","Name","Ticket","PassengerId"],axis=1)
test=test_raw.drop(["Cabin", "Name","Ticket","PassengerId"],axis=1)

#Taking care of missing values
from sklearn.base import TransformerMixin

class DataFrameImputer(TransformerMixin):
    
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

train = pd.DataFrame(train)
train = DataFrameImputer().fit_transform(train)
test = pd.DataFrame(test)
test = DataFrameImputer().fit_transform(test)

#Relation between Passenger class and survived
sns.countplot(data=train, x='Survived', hue='Pclass', palette='Set2')
#relation b/n sex and survived
sns.countplot(data=train, x='Survived', hue='Sex', palette='Set2')
#Relation between age and survived
sns.violinplot(x="Survived", y="Age", data=train)
#relationship between no. of siblings and survived
sns.swarmplot(x="Survived", y="SibSp", data=train)
#Relationship between Parents and survived
sns.swarmplot(x="Survived", y="Parch", data=train)
#Relationship between  relatives present and person survived or not
relatives = train["SibSp"] + train["Parch"]
for n,i in enumerate(relatives):
    if i>0:
        relatives[n]="yes"
    else:
        relatives[n]="no"

relatives1=pd.concat([train["Survived"],relatives],axis=1)

sns.countplot(data=relatives1, x=0, hue="Survived", palette='Set2')
plt.xlabel("Relatives present or not")

#No.of passengers according to fare
m=train["Fare"].mean()
fare_sur= train["Fare"]+0
li=[]
for n,i in enumerate(fare_sur):
    li.append(i)
    if i<=10:
        fare_sur[n]="upto $10"
    elif i>10 and i+1<=40:
        fare_sur[n]="$10-$40"
    elif i>40 and i+1<=80:
        fare_sur[n]="$40-$80"
    else:
        fare_sur[n]=">$80"
fare_sur1=pd.concat([train["Survived"],fare_sur], axis=1)

#Representing no. of passengers accorsing to fair
sns.countplot(data=fare_sur1, x="Fare", palette='Set1')
plt.ylabel("No. of passengers")
#People survived or died in each fair groups
sns.countplot(data=fare_sur1, x="Fare",hue="Survived", palette='Set2')
plt.ylabel("No. of passengers")

#No.of passengers from each port
sns.countplot(data=train, x="Embarked", palette='Set2')
#class of passengers from each port
sns.countplot(data=train, x="Embarked",hue="Pclass", palette='Set2')

￼
#Survivals from each port
sns.countplot(data=train, x="Embarked",hue="Survived", palette='Set2')

#splitting into independent and dependent variables
X = train.iloc[:, train.columns != 'Survived'].values
y = train.iloc[:,0].values
X_test1=test.iloc[:,test.columns!='Survived'].values

#Encoding Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
labelencoder1 = LabelEncoder()
X[:, 1] = labelencoder.fit_transform(X[:, 1])
X[:, 6] = labelencoder.fit_transform(X[:, 6])
X_test1[:, 1]=labelencoder1.fit_transform(X_test1[:, 1])
X_test1[:, 6] = labelencoder1.fit_transform(X_test1[:, 6])
onehotencoder = OneHotEncoder(categorical_features = [6])
onehotencoder1 = OneHotEncoder(categorical_features = [6])
X = onehotencoder.fit_transform(X).toarray()
X_test1 = onehotencoder1.fit_transform(X_test1).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]
X_test1=X_test1[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier(random_state=0)
classifier.fit(X_train,y_train)

"""#Applying logistic regresion algorithm
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X,y)"""

#Predicting the values for test set
y_pred=classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

#Prediction on actualtest set
y_pred1=classifier.predict(X_test1)

#Creating a new CSV file with predicted values
y_pred1=y_pred1.reshape(-1,1)
test_full=np.concatenate((test_raw,y_pred1), axis=1)

df=pd.DataFrame(test_full).set_axis(['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',
               'Ticket', 'Fare', 'Cabin', 'Embarked', 'Survived'], axis=1, inplace=False)
df.to_csv("testcomp1.csv", index=False)




