<<<<<<< HEAD
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
    "account" : [1,2,3,4,5],
    "offline" : [20,30,40,10,24],
    "online"  : [20,30,40,10,24],
    "total" : [40,60,80,20,48]
})

for index, row in df.iterrows():
    print (type(row['account']))
    


Data = [
    	{ "Date": "2016-06-14", "Categories": [{ "Name": "Category1", "Value": 368 }, { "Name": "Category2", "Value": 321 }, { "Name": "Category3", "Value": 524 }], "LineCategory": [{ "Name": "Line1", "Value": 69 }, { "Name": "Line2", "Value": 63 }] },
		{ "Date": "2016-06-20", "Categories": [{ "Name": "Category1", "Value": 412 }, { "Name": "Category2", "Value": 461 }, { "Name": "Category3", "Value": 321 }], "LineCategory": [{ "Name": "Line1", "Value": 75 }, { "Name": "Line2", "Value": 85 }] }
		]

for i,colname in enumerate(df.columns):
    print(i)
    print(colname)
    
#print(df)

#d = [
#    { "Date": "2016-06-14", "Categories": [{ "Name": "Category1", "Value": 368 }, { "Name": "Category2", "Value": 321 }, { "Name": "Category3", "Value": 524 }], "LineCategory": [{ "Name": "Line1", "Value": 69 }, { "Name": "Line2", "Value": 63 }] }
#]

d2 = []
for row in df.values:
    #print(type(row))
    row_l=row.tolist()
    r = { "Account": row_l[0], "Categories": [{ "Name": "offline", "Value": row_l[1] }, { "Name": "online", "Value": row_l[2] }, { "Name": "total", "Value": row_l[3] }], "LineCategory": [{ "Name": "threshold", "Value": 69 }, { "Name": "Line2", "Value": 63 }] }        
    d2.append(dict(r))
#print(d)    
    
#print (json.dumps(d2))    
=======
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
    "account" : [1,2,3,4,5],
    "offline" : [20,30,40,10,24],
    "online"  : [20,30,40,10,24],
    "total" : [40,60,80,20,48]
})

for index, row in df.iterrows():
    print (type(row['account']))
    


Data = [
    	{ "Date": "2016-06-14", "Categories": [{ "Name": "Category1", "Value": 368 }, { "Name": "Category2", "Value": 321 }, { "Name": "Category3", "Value": 524 }], "LineCategory": [{ "Name": "Line1", "Value": 69 }, { "Name": "Line2", "Value": 63 }] },
		{ "Date": "2016-06-20", "Categories": [{ "Name": "Category1", "Value": 412 }, { "Name": "Category2", "Value": 461 }, { "Name": "Category3", "Value": 321 }], "LineCategory": [{ "Name": "Line1", "Value": 75 }, { "Name": "Line2", "Value": 85 }] }
		]

for i,colname in enumerate(df.columns):
    print(i)
    print(colname)
    
#print(df)

#d = [
#    { "Date": "2016-06-14", "Categories": [{ "Name": "Category1", "Value": 368 }, { "Name": "Category2", "Value": 321 }, { "Name": "Category3", "Value": 524 }], "LineCategory": [{ "Name": "Line1", "Value": 69 }, { "Name": "Line2", "Value": 63 }] }
#]

d2 = []
for row in df.values:
    #print(type(row))
    row_l=row.tolist()
    r = { "Account": row_l[0], "Categories": [{ "Name": "offline", "Value": row_l[1] }, { "Name": "online", "Value": row_l[2] }, { "Name": "total", "Value": row_l[3] }], "LineCategory": [{ "Name": "threshold", "Value": 69 }, { "Name": "Line2", "Value": 63 }] }        
    d2.append(dict(r))
#print(d)    
    
#print (json.dumps(d2))    
>>>>>>> df9bbfd486fd1721ef9aa82f490911ce4a35d67d
