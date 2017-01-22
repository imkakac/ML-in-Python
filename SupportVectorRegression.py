# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 11:31:56 2017

@author: Ran
"""
import os
os.getcwd() # Prints the working directory
os.chdir('C:\\Users\\Ran\\OneDrive\\job\\Udemy ML R Python\\SVR')

# Polynomial Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Encoding categorical data
"""from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

sc_y = StandardScaler()
y = sc_y.fit_transform(y)

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf') #object of the class
regressor.fit(X, y)

# Predicting a new result with linear regression
y_pred = regressor.predict(X)
y_pred1 = regressor.predict(6.5)
y_pred2 = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([6.5]))))

# Visualizing the svr results
#X_grid = np.arange(min(X), max(X), 0.1)
#X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
