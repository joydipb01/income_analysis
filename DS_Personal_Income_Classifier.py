# -*- coding: utf-8 -*-
"""
Created on Tue May  5 20:09:40 2020

@author: Joydip
"""

import os
import pandas as pd
import numpy as np
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

os.chdir("C:/Users/Joydip/Documents")
data_income = pd.read_csv("income.csv")
data = data_income.copy()

'''
Exploratory Data Analysis (3 steps):
    1. Getting to know the data
    2. Data Preprocessing (Missing values)
    3. Cross Tables and Data Visualisation
'''

#Getting to know the data

print(data.info())
#missing values:
print(data.isnull().sum())

summary_num = data.describe()
print(summary_num)

summary_cat = data.describe(include="O")
print(summary_cat)

data[' JobType'].value_counts()
data[' occupation'].value_counts()

print(np.unique(data[' JobType']))
print(np.unique(data[' occupation']))

#reading '?' as NA
data = pd.read_csv('income.csv', na_values=[" ?"])

#Data Preprocessing

data.isnull().sum()
missing = data[data.isnull().any(axis=1)] #to consider atleast one column is missing
data2=data.dropna(axis=0)

correlation=data2.corr() #relating independent variables

#Cross Tables and Data Visualisation

data2.columns

gender = pd.crosstab(index=data2[" gender"], columns='count', normalize=True)
print(gender) #Gender proportion table

gender_salstat = pd.crosstab(index=data2[" gender"],
                            columns=data2[" SalStat"],
                            margins=True,
                            normalize='index')
print(gender_salstat)

Salstat=sb.countplot(data2[" SalStat"])
#75% corresponds to <=50K

sb.distplot(data2["age"], bins=10, kde=False)

sb.boxplot(" SalStat", "age", data=data2)
data2.groupby(" SalStat")["age"].median()

sb.countplot(y=' JobType', data=data2, hue=' SalStat')
jobtype_salstat = pd.crosstab(index=data2[" JobType"],
                            columns=data2[" SalStat"],
                            margins=True,
                            normalize='index')
print(jobtype_salstat)
'''
73% State gov employees work at more than 50K. This is an important column for
avoiding misuse of subsidies.
'''

sb.countplot(y=' EdType', data=data2, hue=' SalStat')
edtype_salstat = pd.crosstab(index=data2[" EdType"],
                            columns=data2[" SalStat"],
                            margins=True,
                            normalize='index')
print(edtype_salstat)
'''
Doctorate, Masters and Prof-school graduates are more likely to earn more than
50K. These are influencing variables in avoiding the misuse of subsidy.
'''

sb.countplot(y=' occupation', data=data2, hue=' SalStat')
occ_salstat = pd.crosstab(index=data2[" occupation"],
                            columns=data2[" SalStat"],
                            margins=True,
                            normalize='index')
print(occ_salstat)
'''
Those who make more than 50K are more likely to be exexutive managers and 
professionals, influencing variables in avoiding the misuse of subsidy.
'''

sb.distplot(data2[" capitalgain"], bins=10, kde=False)

sb.distplot(data2[" capitalloss"], bins=10, kde=False)

sb.boxplot(x=data2[" SalStat"], y=data2[" hoursperweek"])
'''
Those who earn more than 50K spend about 40 to 50 hours per week.
This variable can contribute to classifying the individual's salary status,
since there is a clear association between salary status and hours at work per
week.
'''

#LOGISTIC REGRESSION

#Reindexing Salary Status to 0 and 1
data2[" SalStat"] = data2[" SalStat"].map({" <=50K":0, " >50K":1})
print(data2[" SalStat"])

new_data = pd.get_dummies(data2, drop_first=True)

#Storing column names
columns_list = list(new_data.columns)
print(columns_list)

#Separating input names from data
features = list(set(columns_list)-set([" SalStat"]))
print(features)

#Storing output values in y
y = new_data[" SalStat"].values
print(y)

#Storing the values from input features
x = new_data[features].values
print(x)

#Splitting the data into training data and test data
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3,
                                                    random_state=0)

#Making an instance of the model
logistic = LogisticRegression()
#Fitting the values for x and y and training the model
logistic.fit(train_x, train_y)
logistic.coef_
logistic.intercept_

#Prediction from test data
prediction = logistic.predict(test_x)
print(prediction)

#Confusion matrix
confusion_matrix = confusion_matrix(test_y, prediction)
print(confusion_matrix)

#Calculating the accuracy
accuracy_score = accuracy_score(test_y, prediction)
print(accuracy_score)

#Printing missclassified values from the prediction
print("The missclassified samples are %d" % (test_y!=prediction).sum())

#IMPROVING THE MODEL: REMOVING INSIGNIFICANT VARIABLES

cols = [" gender", " nativecountry", " race", " JobType"]
new_data = data2.drop(cols, axis=1)
'''
Now follow the steps from line 128-170
'''

#KNN

from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train_x, train_y)

prediction=knn.predict(test_x)

confusion_matrix=confusion_matrix(test_y, prediction)
print(confusion_matrix)

accuracy_score=accuracy_score(test_y, prediction)
print(accuracy_score)

print("The missclassified samples are %d" % (test_y!=prediction).sum())

#Effect of K on classifier

Misclass_samp=[]

#Calculating misclassified values for nearest neighbors ranging from 1 to 20
for i in range(1, 20):
    KNN = KNeighborsClassifier(n_neighbors=i)
    KNN.fit(train_x, train_y)
    pred_i=KNN.predict(test_x)
    Misclass_samp.append((test_y!=pred_i).sum())
print(Misclass_samp)
