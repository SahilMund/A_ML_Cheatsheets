import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataframe = pd.read_csv("contraceptivemethodchoice.csv")

df = dataframe.copy()

df.set_axis(['W_age','W_ed','H_ed', 'Kids', 'W_reg','W_occ', 'H_occ', 'Sol', 'M_exp','Method_used'], inplace=True, axis=1)
corr=df.corr()
df.describe()
sns.heatmap(corr)
corr=corr['Method_used']
df.drop(df.iloc[:,[0,4,8]], inplace=True, axis=1)
df.isnull().sum()


X=df.iloc[:,df.columns!="Method_used"].values
y=df.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from catboost import CatBoostClassifier
classifier=CatBoostClassifier(iterations=50)
classifier.fit(X_train,y_train,cat_features=[0,1,3,4,5], eval_set=(X_test,y_test), plot=True)

from xgboost import XGBClassifier
classifier1 = XGBClassifier(random_state=0, n_estimators=100)
classifier1.fit(X_train,y_train)

from sklearn.tree import DecisionTreeClassifier
classifier2=DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier2.fit(X_train,y_train)


y_pred = classifier.predict(X_test)
y_pred1 = classifier1.predict(X_test)
y_pred2 = classifier2.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm1 = confusion_matrix(y_test, y_pred1)
cm2 = confusion_matrix(y_test, y_pred2)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

accuracies1 = cross_val_score(estimator = classifier1, X = X_train, y = y_train, cv = 10)
accuracies1.mean()
accuracies1.std()

accuracies2 = cross_val_score(estimator = classifier2, X = X_train, y = y_train, cv = 21)
accuracies2.mean()
accuracies2.std()


