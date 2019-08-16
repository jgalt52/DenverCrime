# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Author Hunt Waggoner
#Purpose To look at the denver crime data and see what I can figure out
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Lets import our Data Set
dataset = pd.read_csv('crime.csv')
#My first idea is to see if there is a correlation between the type of crime
#and the district id which is 15 index 14 in python
#So we need Offense Type which is col 6 index 5 in python
X = dataset.iloc[:,5:6].values
y = dataset.iloc[:,14].values


#Now since our data in offense category is words
#lets categorize it
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn import preprocessing
labelencoder = LabelEncoder()
X = labelencoder.fit_transform(X)

#Ok our categories are now numerical
#lets make them into their own columns of 0 and 1
onehotencoder = OneHotEncoder(categorical_features =[0], n_values ='auto')
X=X.reshape(-1,1)
X = onehotencoder.fit_transform(X).toarray()

#Ok we got our categories into their own data columns
#Lets deal with the dummy variable trap
X = X[:,1:]

#SPlit the data into training and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state = 0)

#lets Fit our data to the multiple linear regression training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predict some test sets
y_pred = regressor.predict(X_test)
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf

Xz = sm.add_constant(X)

X =np.append(arr = np.ones((504611,1)).astype(int), values = X , axis = 1)
X_opt = X[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]]
regressor_OLS = smf.ols(data = X_opt, formula = Xz).fit()
regressor_OLS.summary()


