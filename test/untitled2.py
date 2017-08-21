# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 20:55:27 2017

@author: SheikhRabiul
"""

from flask import Flask, render_template, Response,make_response

import random
import json
import pandas
import numpy as np
 
df = pandas.DataFrame({
    "x" : [11,28,388,400,420],
    "y" : np.random.rand(5)
})

for i,colname in enumerate(df.columns):
    print(i)
    print(colname)
    
print(df)

d = [
    dict([
        (colname, row[i])
        for i,colname in enumerate(df.columns)
    ])
    for row in df.values
]
    
#print(d)    
    
print (json.dumps(d))    
