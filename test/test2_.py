#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#http://blog.datadive.net/selecting-good-features-part-iii-random-forests/
"""
Created on Tue Aug  8 01:09:01 2017

@author: sheikh
"""

from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('dataset_olap_mini.csv')
#print(dataset.head())
X = dataset.iloc[:,1:-1].values
X_columns = dataset.iloc[:,1:-1].columns.values
y = dataset.iloc[:,-1].values
# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
X[:, 5] = labelencoder_X.fit_transform(X[:, 5])
X[:, 6] = labelencoder_X.fit_transform(X[:, 6])
X[:, 8] = labelencoder_X.fit_transform(X[:, 8])
X[:, 9] = labelencoder_X.fit_transform(X[:, 9])
X[:, 11] = labelencoder_X.fit_transform(X[:, 11])
X[:, 13] = labelencoder_X.fit_transform(X[:, 13])
X[:, 14] = labelencoder_X.fit_transform(X[:, 14])
X[:, 16] = labelencoder_X.fit_transform(X[:, 16])
X[:, 18] = labelencoder_X.fit_transform(X[:, 18])
X[:, 19] = labelencoder_X.fit_transform(X[:, 19])

onehotencoder = OneHotEncoder(categorical_features = [0,2,3,5,6,8,9,11,13,14,16,18,19])
X = onehotencoder.fit_transform(X).toarray()
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)

indices = np.argsort(importances)[::-1]
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


print ("Features sorted by their score:")
print (sorted(zip(map(lambda x: round(x, 4), forest.feature_importances_), X_columns), 
             reverse=True))