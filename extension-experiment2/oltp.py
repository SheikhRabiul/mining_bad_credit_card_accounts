# Author: Sheikh Rabiul Islam
# Date: 02/10/2018
# Purpose: Predicting credit card clients with default on payment; processing first batch of OLTP data

#import modules
import pandas as pd   
import sqlite3
import time
start = time.time()
#db connection to common db
conn = sqlite3.connect("credit.sqlite")
curr = conn.cursor()
serial = 5; 
#grab all accounts with corresponding risk probability
table_name = "result_" + str(serial)
df_result = pd.read_sql_query("select * from " + table_name + ";",conn)
conn.commit()
curr.close()
conn.close()

lmda = .5

# import OLTP data
database = "credit" + str(serial) + ".sqlite"
conn = sqlite3.connect(database)
curr = conn.cursor()

table_name = "oltp" + str(serial)
for index, row in df_result.iterrows():
    r_online = 0
    r_offline = 0
    r_total= 0
    
    #process expenditure type transactions of this batch
    df_sum_amt = pd.read_sql_query("select sum(amount) as sum_amount from " + table_name + " where account=:param and type=:param2;",conn, params={'param':row['account'], 'param2':'exp'})
    sum_amt = df_sum_amt.iat[0,0]
    if not sum_amt is None:
        #
        #if running sum of transaction for an account is greater than balance_limit then risky
        if sum_amt > row['balance_limit']:
            r_online = lmda
        
    # process payment type transactions of this batch
    df_sum_pay_amt = pd.read_sql_query("select sum(amount) as sum_pay_amount from " + table_name + " where account=:param and type=:param2;",conn, params={'param':row['account'], 'param2':'pay'})
    sum_pay_amt = df_sum_pay_amt.iat[0,0]
    # check payment amount of this month + previous total against total dues and repayment
    gap = row['total_bill'] - ( row['total_payment'] + sum_pay_amt) 
    
    if gap > row['balance_limit'] and row['repayment'] >=1:  # didn't even paid the minimum due amount in last # month
        #repayment status is: -1 = pay duly; 0 = payment less than bill; -2 = no bill but there is payment;  1 = payment delay for one month; 
        #2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above.
        r_online = lmda
    
    if row['online'] < r_online: # if online risk probability already high than this then keep it same 
        #
        df_result.at[index,'online'] = r_online
        #total risk probability should not exceed 1.
        tot = r_online + row['offline']
        if tot < 1:
            df_result.at[index,'offline'] = tot
        else:
            df_result.at[index,'offline'] = 1
        # make decision
        if tot > .5:
            df_result.at[index,'default'] = 1
    
conn.commit()
curr.close()
conn.close()
            
    
conn = sqlite3.connect("credit.sqlite")
curr = conn.cursor()
#Store the updated result in to database
table_name = "result_" + str(serial)
df_result.pop('index')
df_result.to_sql(table_name, conn, if_exists="replace")

conn.commit()
curr.close()
conn.close()

end = time.time()
diff = end - start
print("time taken: ")
print(diff)
