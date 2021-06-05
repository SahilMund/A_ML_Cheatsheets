import pandas as pd
import numpy as np
import seaborn as sns

#setting dimensions for plot

sns.set(rc={'figure.figsize':(11.7,8.27)})
 
#reading csv file
cars_data=pd.read_csv('cars_sampled.csv')

#creating a copy
cars=cars_data.copy()

cars.info()

cars.describe()
pd.set_option('display.float_format', lambda x: '%.3f' % x)
cars.describe()
pd.set_option('display.max_columns', 500)
cars.describe()

#dropping unwanted cols
cars=cars.drop(columns=['name', 'dateCrawled','dateCreated', 'postalCode', 'lastSeen'], axis=1)

#removing duplicate records
cars.drop_duplicates(keep='first', inplace=True)

#finding no. of missing values
cars.isnull().sum()

#plotting year of reg wrt to price
yearwise_count=cars['yearOfRegistration'].value_counts().sort_index()
sum(cars['yearOfRegistration']>2018) #no.of cars registered after 2k18 (ie in future)
sum(cars['yearOfRegistration']<1950) #no.of cars registered after 1950 (ie in extreme past)
sns.regplot(x='yearOfRegistration', y='price', scatter=True, fit_reg=False, data=cars)
#working range between 1950 and 2018

#plotting year of reg wrt to price
price_count=cars['price'].value_counts().sort_index()
sns.distplot(cars['price']) #it shows the frequency of cars being sold for a particular price.
cars['price'].describe()
sns.boxplot(y=cars['price'])

sum(cars['price']>150000) #no.of cars registered above 150000
sum(cars['price']<100) #no.of cars registered below  100
#workuing range bn 100 to 150000

#variable powerPS
power_count=cars['powerPS'].value_counts().sort_index()
sns.distplot(cars['powerPS'])
cars['powerPS'].describe()
sns.boxplot(y=cars['powerPS'])
sns.regplot(x='powerPS', y='price', scatter=True, fit_reg=False, data=cars)
sum(cars['powerPS']>500)
sum(cars['powerPS']<10)
#working range 10 and 500

"""Part Two"""

#working range of data
cars=cars[(cars.yearOfRegistration<=2018)
          &(cars.yearOfRegistration>=1950)
          &(cars.price>=100)
          &(cars.price<=150000)
          &(cars.powerPS>=10)
          &(cars.powerPS<=500)]

#Combining yr of reg. and month of reg. To find the total age of the car
cars['monthOfRegistration']/=12

#creating new variable age by adding yr of reg and month of reg
cars['Age']=(2018-cars['yearOfRegistration'])+(cars['monthOfRegistration'])

#rounding off the age values
cars['Age']= round(cars['Age'],2)
cars['Age'].describe()

#dropping year of reg and month of reg as it is converted to age

cars=cars.drop(columns=['yearOfRegistration', 'monthOfRegistration'], axis=1)

"""VISUALIZATION"""

#AGE
sns.distplot(cars['Age'])
sns.boxplot(y=cars['Age'])

#PRICE
sns.distplot(cars['price'])
sns.boxplot(y=cars['price'])

#powerPS
sns.distplot(cars['powerPS'])
sns.boxplot(y=cars['powerPS'])

#visualizing parameters after narrowing working range

#Age vs Price
sns.regplot(x='Age', y='price', scatter=True, fit_reg=False, data=cars)
#cars priced higher are newer
#with increase in age, price decreases
#however some cars are priced higher with increase in age

#PowerPS vs Price
sns.regplot(x='powerPS', y='price', scatter=True, fit_reg=False, data=cars)
#with increase in power price increases

#variable seller
cars['seller'].value_counts()
pd.crosstab(cars['seller'], columns='count', normalize=True)
sns.countplot(x='seller', data=cars)
#fewer cars have commercial i.e insignificant

#variable OfferType
cars['offerType'].value_counts()
pd.crosstab(cars['offerType'], columns='count', normalize=True)
sns.countplot(x='offerType', data=cars)
#all cars have offer, so it is insignificant

#variable AB test
cars['abtest'].value_counts()
pd.crosstab(cars['abtest'], columns='count', normalize=True)
sns.countplot(x='abtest', data=cars)
#both are almost equally distributed

sns.boxplot(x='abtest', y='price', data=cars)
# for every price value, there is almost 50- 50 distribution
#dostnt affect price, insignificant

#variable vehicleType
cars['vehicleType'].value_counts()
pd.crosstab(cars['vehicleType'], columns='count', normalize=True)
sns.countplot(x='vehicleType', data=cars)
sns.boxplot(x='vehicleType', y='price', data=cars)
#vechile type affects price, significant

#variable gearbox
cars['gearbox'].value_counts()
pd.crosstab(cars['gearbox'], columns='count', normalize=True)
sns.countplot(x='gearbox', data=cars)
sns.boxplot(x='gearbox', y='price', data=cars)
#gearboc affects price, significant

#variable model
cars['model'].value_counts()
pd.crosstab(cars['model'], columns='count', normalize=True)
sns.countplot(x='model', data=cars)
sns.boxplot(x='model', y='price', data=cars)
#cars are distributed over many models
#considered in modelling

#variable kilometers
cars['kilometer'].value_counts().sort_index()
pd.crosstab(cars['kilometer'], columns='count', normalize=True)
sns.countplot(x='kilometer', data=cars)
sns.boxplot(x='kilometer', y='price', data=cars)
sns.distplot(cars['kilometer'], bins= 8, kde=False)
sns.regplot(x='kilometer', y='price', scatter=True, fit_reg=True, data=cars)
cars['kilometer'].describe()
#considered in modelling

#variable fuelType
cars['fuelType'].value_counts()
pd.crosstab(cars['fuelType'], columns='count', normalize=True)
sns.countplot(x='fuelType', data=cars)
sns.boxplot(x='fuelType', y='price', data=cars)
#fuel type effects price, ie significant

#variable brand
cars['brand'].value_counts()
pd.crosstab(cars['brand'], columns='count', normalize=True)
sns.countplot(x='brand', data=cars)
sns.boxplot(x='brand', y='price', data=cars)
#cars are distributed over many brands, so considered


#variable notRepairedDamage
#yes-car is damaged but not rectified
#no- car was damaged but has been rectified
cars['notRepairedDamage'].value_counts()
pd.crosstab(cars['notRepairedDamage'], columns='count', normalize=True)
sns.countplot(x='notRepairedDamage', data=cars)
sns.boxplot(x='notRepairedDamage', y='price', data=cars)
#AS EXPECTED THE CARS THAT require the damages to be repaired
#fail under lower price ranges

"""REMOVING INSIGNIFICANT VARIABLES"""
col=['seller', 'offerType', 'abtest']
cars=cars.drop(columns=col, axis=1)
cars_copy=cars.copy()

"""CORELATION"""
cars_select1=cars.select_dtypes(exclude=[object])
correlation=cars_select1.corr()
round(correlation,3)

cars_select1.corr().loc[:,'price'].abs().sort_values(ascending=False)[1:]

#omitting missing values
cars_omit=cars.dropna(axis=0)

#converting categorical variables to dummy variables
cars_omit=pd.get_dummies(cars_omit, drop_first=True)

#importing necessary libraries

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

"""Building Model With Omitted Data"""

#Separating input and output features
x1=cars_omit.drop(['price'], axis='columns', inplace=False)
y1=cars_omit['price']

#plotting the variable price 
prices=pd.DataFrame({"1.Before":y1, "2.After":np.log(y1)})
prices.hist()

#transforming the price as a logarithmic value.
y1=np.log(y1)

#splitting data into test and train set
X_train, X_test, y_train, y_test=train_test_split(x1,y1,test_size=0.3,random_state=3)
print(X_train.shape, X_test.shape, y_train.shape,y_test.shape)

#finding the mean for test data value
base_pred=np.mean(y_test)
print(base_pred)

#repeating the same value titll length of the test data
base_pred=np.repeat(base_pred, len(y_test))
bp=pd.DataFrame(base_pred)

#finding RMSE
base_root_mean_square_error=np.sqrt(mean_squared_error(y_test, base_pred))

"""LINEAR REGRESSION"""

#sETTING INTERCEPT AS TRUE
lgr=LinearRegression(fit_intercept=True)
#model
model_lin1=lgr.fit(X_train,y_train)

#predicting model on test set
cars_predictions_lin1=lgr.predict(X_test)

#computing MSE and RMSE
lin_mse1=mean_squared_error(y_test,cars_predictions_lin1)
lin_rmse1=np.sqrt(lin_mse1)

print(lin_rmse1)

#Rsquared value
r2_lin_test1=model_lin1.score(X_test,y_test)
r2_lin_train1=model_lin1.score(X_train,y_train)
print(r2_lin_test1,r2_lin_train1)

#regression diagnostic-residual plot analsis
residuals1=y_test-cars_predictions_lin1
sns.regplot(x=cars_predictions_lin1,y=residuals1, scatter=True, fit_reg=False)
residuals1.describe()


#random forest with omitted data

rf=RandomForestRegressor(n_estimators=100, max_features='auto', max_depth=100,
                         min_samples_split=10, min_samples_leaf=4, random_state=1)

model_rf1=rf.fit(X_train, y_train) 

#predicting model on test set
cars_predictions_rf1=rf.predict(X_test)

#computing MSE and RMSE
rf_mse1=mean_squared_error(y_test,cars_predictions_rf1)
rf_rmse1=np.sqrt(rf_mse1)
print(rf_rmse1)

#Rsquared value
r2_rf_test1=model_rf1.score(X_test,y_test)
r2_rf_train1=model_rf1.score(X_train,y_train)
print(r2_rf_test1,r2_rf_train1)

"""MODEL BUILDING WITH IMPUTED DATA"""

cars_imputed = cars.apply(lambda x:x.fillna(x.median())\
        if x.dtype=='float' else\
        x.fillna(x.value_counts().index[0]))

cars_imputed.isnull().sum()

#converting categorical variable to dummy variables
cars_imputed=pd.get_dummies(cars_imputed, drop_first=True)

#separating input and output features
x2=cars_imputed.drop(['price'], axis='columns', inplace=False)
y2=cars_imputed['price']

#plotting the variable price 
prices=pd.DataFrame({"1.Before":y2, "2.After":np.log(y2)})
prices.hist()

#transforming the price as a logarithmic value.
y2=np.log(y2)

#splitting data into test and train set
X_train1, X_test1, y_train1, y_test1=train_test_split(x2,y2,test_size=0.3,random_state=3)
print(X_train1.shape, X_test1.shape, y_train1.shape,y_test1.shape)

#base line model for imputed data
#we are making a base model by using test data mean value. this is to set a bench mark and to compare with our regression model

#finding the mean for test data value
base_pred = np.mean(y_test1)
print(base_pred)

#repeating same value till length of the test data
base_pred=np.repeat(base_pred,len(y_test1))

#finding RMSE
base_root_mean_square_error_imputed=np.sqrt(mean_squared_error(y_test1, base_pred))
print(base_root_mean_square_error_imputed)

"""LINEAR REGRESSION WITH IMPUTED DATA"""

#Setting intercept as true
lgr2=LinearRegression(fit_intercept=True)

#model
model_lin2=lgr2.fit(X_train1, y_train1)

#predicting model on test set
cars_predictions_lin2=lgr2.predict(X_test1)

#computing MSE and RMSE
lin_mse2=mean_squared_error(y_test1,cars_predictions_lin2)
lin_rmse2=np.sqrt(lin_mse2)

print(lin_rmse2)

#Rsquared value
r2_lin_train2=model_lin2.score(X_test1,y_test1)
r2_lin_test2=model_lin2.score(X_train1,y_train1)
print(r2_lin_test2,r2_lin_train2)

"""RANDOM FOREST WITH IMPUTED DATA"""

#MODEL PARAMETERS
rf2=RandomForestRegressor(n_estimators=100, max_features='auto', max_depth=100,
                         min_samples_split=10, min_samples_leaf=4, random_state=1)

model_rf2=rf2.fit(X_train1, y_train1) 

#predicting model on test set
cars_predictions_rf2=rf2.predict(X_test1)

#computing MSE and RMSE
rf_mse2=mean_squared_error(y_test1,cars_predictions_rf2)
rf_rmse2=np.sqrt(rf_mse2)
print(rf_rmse2)

#Rsquared value
r2_rf_test2=model_rf2.score(X_test1,y_test1)
r2_rf_train2=model_rf2.score(X_train1,y_train1)
print(r2_rf_test2,r2_rf_train2)

"""KHATAAAAM"""