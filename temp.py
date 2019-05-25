#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:23:25 2019

@author: Ozan
"""
#import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt
#import pandas as pd
#from sklearn.model_selection import train_test_split
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.tree import DecisionTreeClassifier
#
#dataset = pd.read_csv('5000.csv')
#print(dataset.columns)
#dataset.head()
#print("Dimension of Sales data: {}".format(dataset.shape))
#print(dataset.groupby('Sales Channel').size())
#sns.countplot(dataset['Sales Channel'],label="Count")
#dataset.info()
#
#X_train, X_test, y_train, y_test = train_test_split(dataset.loc[:, dataset.columns != 'Sales Channel'], dataset['Sales Channel'], stratify=dataset['Sales Channel'], random_state=66)
#training_accuracy = []
#test_accuracy = []
#tree = DecisionTreeClassifier(random_state=0)
#tree.fit(X_train, y_train)
#print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
#print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))
#
## try n_neighbors from 1 to 10
#neighbors_settings = range(1, 11)
#for n_neighbors in neighbors_settings:
#    # build the model
#    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
#    knn.fit(X_train, y_train)
#    # record training set accuracy
#    training_accuracy.append(knn.score(X_train, y_train))
#    # record test set accuracy
#    test_accuracy.append(knn.score(X_test, y_test))
#plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
#plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
#plt.ylabel("Accuracy")
#plt.xlabel("n_neighbors")
#plt.legend()
#plt.savefig('knn_compare_model')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv('newFile.csv')
# We are using Online as 1, Offline as 0
#In this code i want to use columns that i decided use in program.
#keep_col = ['Units Sold','Unit Price','Unit Cost','Total Revenue', 'Total Cost', 'Total Profit', 'Sales Channel']
#new_f = df[keep_col]
#new_f.to_csv("newFile.csv", index=False)
print(df.head())
print(df.columns)
print(df.dtypes)
print(df.describe())
print(df.groupby('Sales Channel').size())
#We are going to use Onlines as 1, Offlines as 0. I've changed it because there is a
#problem about string to float conversion.
sns.countplot(df['Sales Channel'],label="Count")
print(df.info())

X_train, X_test, y_train, y_test = train_test_split(df.loc[:, df.columns != 'Sales Channel'],
                                                    df['Sales Channel'], stratify=df['Sales Channel'], random_state=66)
training_accuracy = []
test_accuracy = []
# try n_neighbors from 1 to 10
neighbors_settings = range(1, 8)
for n_neighbors in neighbors_settings:
    # build the model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    # record training set accuracy
    training_accuracy.append(knn.score(X_train, y_train))
    # record test set accuracy
    test_accuracy.append(knn.score(X_test, y_test))
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.savefig('knn_compare_model')

knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'.format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'.format(knn.score(X_test, y_test)))

tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print("Accuracy on tree decision training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on tree decision test set: {:.3f}".format(tree.score(X_test, y_test)))

tree = DecisionTreeClassifier(max_depth=2, random_state=0)
tree.fit(X_train, y_train)
print("Accuracy on Depth 2 tree decision training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on Depth 2 tree decision test set: {:.3f}".format(tree.score(X_test, y_test)))

tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)
print("Accuracy on Depth 4 tree decision training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on Depth 4 tree decision test set: {:.3f}".format(tree.score(X_test, y_test)))

tree = DecisionTreeClassifier(max_depth=6, random_state=0)
tree.fit(X_train, y_train)
print("Accuracy on Depth 6 tree decision training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on Depth 6 tree decision test set: {:.3f}".format(tree.score(X_test, y_test)))

print("Feature importances:\n{}".format(tree.feature_importances_))

