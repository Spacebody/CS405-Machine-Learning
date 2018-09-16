#!/usr/bin/env python
# coding: utf-8

# Author: Yilin Zheng

# import necessary libraries
import numpy as np
import pandas as pd
from time import time
from IPython.display import display

# visuals.py is a supplementary for data visualization
import visuals as vs

get_ipython().run_line_magic('matplotlib', 'inline')

data = pd.read_csv("census.csv")

display(data.head(n=5))

income_raw = data['income'].copy()
features_raw = data.drop('income', axis=1)

vs.distribution(data)

skewed = ['capital-gain', 'capital-loss']
feature_log_transformed = pd.DataFrame(data = features_raw)
feature_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))

vs.distribution(feature_log_transformed, transformed=True)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

features_log_minmax_transform = pd.DataFrame(data=feature_log_transformed)
features_log_minmax_transform[numerical] = scaler.fit_transform(feature_log_transformed[numerical])

display(features_log_minmax_transform.head(n=5))

# Exercise 0
n_records = len(data.index)
display(n_records)

n_greater_50K = len(data[data['income']==">50K"])
display(n_greater_50K)

n_at_most_50K = len(data[data['income']=="<=50K"])
display(n_at_most_50K)

greater_percent = n_greater_50K/n_records
display(greater_percent)

for col in data.columns:
    display(data[col].unique())

# Exercise 1
# use one hot encoder to encode non-numeric features
non_numeric = ['workclass', 'education_level', 'marital-status', 'occupation', 
               'relationship', 'race', 'sex', 'native-country']
features_non_numeric = data[non_numeric]
features_non_numeric_one_hot = pd.get_dummies(features_non_numeric)
display(features_non_numeric_one_hot.head(n=5))

# convert income_raw, <=50K: 0, >50K: 1

income_raw[income_raw=='<=50K'] = 0
income_raw[income_raw=='>50K'] = 1
display(income_raw.head(n=10))

income = income_raw.tolist()

from sklearn.model_selection import train_test_split

features_final = data.copy()

features_final = features_final.drop(columns=non_numeric)
features_final = features_final.drop(columns=['income'])
features_final[numerical] = features_log_minmax_transform[numerical]
features_final[features_non_numeric_one_hot.columns] = features_non_numeric_one_hot
# features_final['income'] = income
display(features_final.head(n=10))

# Spliting data into 80% training data and 20% test data
X_train, X_test, y_train, y_test = train_test_split(features_final, income, test_size=0.2, shuffle=True)

print("X train has {} sample".format(X_train.shape[0]))
print("X test has {} sample".format(X_test.shape[0]))
print("y train has {} sample".format(len(y_train)))
print("y test has {} sample".format(len(y_test)))


# Exercise 2
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# decision tree has the best accuracy, here we choose it.
model = DecisionTreeClassifier(criterion='entropy')
# SVC has no outperformance result but take longer time to train the data
# model = SVC()
# Guassian NB has the best recall score
# model = GaussianNB()

model = model.fit(X_train, y_train)
# Prediction
y_pred = model.predict(X_test)


# Evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
# bete = 0.5
fbeta = fbeta_score(y_test, y_pred, beta=0.5)

print("Accuracy: {:.3f}".format(accuracy))

print("Precision: {:.3f}".format(precision))

print("Recall: {:.3f}".format(recall))

print("F-beta: {:.3f}".format(fbeta))




