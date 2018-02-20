# -*- coding: utf-8 -*-
"""
Author: Sheikh Rabiul Islam
Date: 02/08/2018
Purpose: Break down total amount into individual transactions following some distribution.
"""
import pandas as pd
import numpy as np
import sqlite3

# Max, Min ranges
max_taiwan = 1684259
min_taiwan = 0
max_spain = 28962.36
min_spain =0

# normalization function
def scale_a_number(inpt, to_min, to_max, from_min, from_max):
    return (to_max-to_min)*(inpt-from_min)/(from_max-from_min)+to_min
# sqlite3 in memmory database for faster query
#conn = sqlite3.connect("credit.sqlite")
conn = sqlite3.connect(":memory:")
conn.isolation_level = None
curr = conn.cursor()

# read a OLTP dataset
df_oltp_whole = pd.read_csv('dataset_OLTP1_mini.csv', sep=',')

# separate transactions of type exp or total bill
df_oltp_filtered = df_oltp_whole[df_oltp_whole['type'] == 'exp'] 

# separate transactions of type pay or payment
df_oltp_filtered2 = df_oltp_whole[df_oltp_whole['type'] == 'pay'] 

#read distribution range dataset; this dataset is made from spanish dataset, it contains all individual transactions 
#amount	total
#35.13	239.93
#27.63	239.93
#13.46	239.93
#28.86	239.93
#64.99	239.93
#33.98	239.93
#35.88	239.93

df_data_distribution = pd.read_csv('data_distribution.csv')

df_data_distribution.to_sql("data_distribution", conn, if_exists="replace")
curr.execute("create index idx_total on data_distribution(total);")
conn.commit()

# unique values in distribution
range_v = sorted(df_data_distribution.total.unique())
# append lowest value (0) of range 
range_v.insert(0,0.0)

# calculate range with upper and lower limit
tp_range = pd.qcut(range_v, len(range_v),precision=2,retbins=True, labels=False)
#tp_range = pd.qcut([0, 1, 3, 7,8,9], 6, precision=2,retbins=True, labels=False)
np_range = tp_range[1]

df_ranges = pd.DataFrame(columns = ['amount', 'lower_bound', 'upper_bound'])

df_oltp_new = pd.DataFrame(columns = ['tid', 'account', 'amount', 'date', 'type'])

# save all total amount and corresponding upper and lower bound
for i in range(len(range_v)):
    row_dict = {'amount': range_v[i], 'lower_bound': np_range[i], 'upper_bound': np_range[i+1]}
    df_ranges = df_ranges.append(row_dict,ignore_index=True)

#insert ranges in to database and index them for faster query     
df_ranges.to_sql("ranges", conn, if_exists="replace")
curr.execute("create index idx_low_upper on ranges(lower_bound,upper_bound);")
conn.commit()

tid = 0


# device individual transactions from total amount
for i in range(len(df_oltp_filtered)):
    #normalize data to spanish dataset range
    account = df_oltp_filtered.loc[i, 'account']
    amount = df_oltp_filtered.loc[i, 'amount']
    date = df_oltp_filtered.loc[i, 'date']
    t_type = df_oltp_filtered.loc[i, 'type']
    amount_nor = scale_a_number(amount, min_spain, max_spain, min_taiwan, max_taiwan)
    #range_row = df_ranges.iloc[np.where((df_ranges.lower_bound.values <= amount_nor) & (df_ranges.upper_bound.values > amount_nor) )]
    #range_row_dicts = range_row.to_dict(orient='records')
    #sql = "select * from ranges where lower_bound <= " + amount_nor +  " and upper_bound > " + amount_nor + ";"
    range_row = pd.read_sql_query("select * from ranges where lower_bound <= :param and upper_bound >:param;",conn, params={'param':amount_nor})
    
    amount_r = 0
    #if(len(range_row_dicts)>0):    
    # #   data_row_dict = range_row_dicts[0]
    #  #  amount_r = data_row_dict['amount']
    
    #calculate coeff; this is needed to denormalize the data into original form.
    for index, row in range_row.iterrows():
        amount_r = row['amount']
        
    coeff = 1
    if amount_r > 0:    
        coeff = amount/amount_r;
    
    #find distribution for the amount
    #df_data_distribution_selected = df_data_distribution[df_data_distribution['total'] == round(amount_r,2)] 
    df_data_distribution_selected = pd.read_sql_query("select * from data_distribution where total = :param;",conn, params={'param':round(amount_r,2)})
    for index, row in df_data_distribution_selected.iterrows():
        amt = row['amount']*coeff # denormalization
        row_dict = {'tid': tid, 'account': account, 'amount': amt,'date': date,'type': t_type}
        df_oltp_new = df_oltp_new.append(row_dict,ignore_index=True)
        tid = tid +1

curr.close()
conn.close()
# append payment transactions at the end of new OLTP dtaset
for index, row in df_oltp_filtered2.iterrows():
    row_dict = {'tid': row['tid'], 'account': row['account'], 'amount': row['amount'],'date': row['date'],'type': row['type']}
    df_oltp_new = df_oltp_new.append(row_dict,ignore_index=True)
    tid = tid +1

conn = sqlite3.connect("credit1.sqlite")
curr = conn.cursor()
    
df_oltp_new.to_sql("oltp1", conn, if_exists="append")
conn.commit()
curr.close()
conn.close()
#writing generated OLTP data in csv file
#file_name = 'OLTP1.csv'
#df_oltp_new.to_csv(file_name,encoding='utf-8')
