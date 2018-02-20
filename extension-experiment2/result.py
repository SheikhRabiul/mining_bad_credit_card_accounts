# Author: Sheikh Rabiul Islam
# Date: 02/10/2018
# Purpose: Predicting credit card clients with default on payment

#import modules
import pandas as pd   
import numpy as np
import sqlite3


conn = sqlite3.connect("credit.sqlite")
curr = conn.cursor()
serial = 4
table_name = "result_" + str(serial)
df_result = pd.read_sql_query("select * from " + table_name + ";",conn)
conn.commit()
curr.close()
conn.close()

y = df_result['default_actual']
y_pred_all = df_result['default']

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score
from sklearn.metrics import precision_recall_fscore_support
cm = confusion_matrix(y, y_pred_all)

#accuracy -number of instance correctly classified
acsc = accuracy_score(y, y_pred_all) 
df_cm = pd.DataFrame([[cm[1][1], cm[0][0],cm[0][1], cm[1][0]]], 
                        index=[0],
                        columns=['True Positives','True Negatives', 'False Positives', 'False Negatives'])

#precision, recall, fscore, support
precision, recall, fscore, support = precision_recall_fscore_support(y, y_pred_all, average='binary')

df_metrics = pd.DataFrame([[acsc, precision, recall, fscore]], 
                        index=[0],
                        columns=['accuracy','precision', 'recall', 'fscore'])


