#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 02:23:40 2017

@author: sheikh
"""

import pandas as pd
import numpy as np
df = pd.DataFrame({'id' : range(1,9),
                'B' : ['one', 'one', 'two', 'three',
                       'two', 'three', 'one', 'two'],
                'amount' : range(1,9)})
print(df)


#select row with id 7

print(df.loc[df['id'] == 7])




#update B='two-five' where id=5
df.loc[df['id']==5,'B'] ='two-five'
print(df)


#using numpy
#update B='theree four' where id=4
df['B'] = np.where(df['id']==4,'three four','zero')
print(df)

#using numpy is faster for selection ref: https://stackoverflow.com/questions/17071871/select-rows-from-a-dataframe-based-on-values-in-a-column-in-pandas
#selecting the result/rows where id=2
print(df.iloc[np.where(df.id.values==2)])


#converiting selected rows in to a dictionary
df2=df.iloc[np.where(df.id.values==2)]
mydict = df2.to_dict(orient='records') 
print(mydict)
mydict2=mydict[0]
print(mydict2['amount'])

# python builtin data types -> list
l = list()
l = ["first element"]

#l+= ["second element"] or
l.append("second element")
print(l)


# concatenating elements of a list
sentence = ['this','is','a','sentence']
sentence_str = '\n'.join(sentence)
print(sentence_str)


