#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 14:03:14 2017

@author: sheikh
"""
import pandas as pd
import numpy as np

df1 = pd.DataFrame([[1,2],[2,3],[3,4]], columns = ['account','bal'])
df2 = pd.DataFrame([[2,5],[3,6],[4,7]], columns = ['account','bal'])

df3 = df1.append(df2, ignore_index=True)

#drop duplicates
df4 = df3.drop_duplicates(subset = ['account'], keep = 'first')

#append newly added results in to result_oltp
result_oltp = pd.DataFrame([[1,20,0,0,None],[2,20,0,0,None]], columns = ['account','offline','online','total','event'])
df2_new = df2['account']
print(type(df2_new))
df2_new = df2_new.to_frame()
print(type(df2_new))
df2_new['offline']=0
df2_new['online']=0
df2_new['total']=0
df2_new['event']=None

df5 = result_oltp.append(df2_new, ignore_index=True)
df6 = df5.drop_duplicates(subset = ['account'], keep = 'first')