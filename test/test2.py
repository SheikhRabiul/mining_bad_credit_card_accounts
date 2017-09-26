#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#http://blog.datadive.net/selecting-good-features-part-iii-random-forests/
"""
Created on Tue Aug  8 01:09:01 2017

@author: sheikh
"""

from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
import numpy as np
#Load boston housing dataset as an example
boston = load_boston()
X = boston["data"]
Y = boston["target"]
names = boston["feature_names"]
rf = RandomForestRegressor()
rf.fit(X, Y)
print ("Features sorted by their score:")
print (sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), 
             reverse=True))
