# Author: Sheikh Rabiul Islam
# Date: 02/10/2018
# Purpose: Predicting credit card clients with default on payment

#import modules
import pandas as pd   
import numpy as np
import sqlite3
import time


# import data
serial = 5
file_name = "dataset_OLAP" + str(serial) + ".csv"
dataset = pd.read_csv(file_name, sep=',')
dataset_slice =  dataset[['total_bill', 'total_payment','repayment', 'balance_limit']]
print("\n imported olap data \n")
#print(dataset.sample(5))  #shows random 5 rows

# seperate the dependent (target) variaable
X = dataset.iloc[:,1:-1].values
X_columns = dataset.iloc[:,1:-1].columns.values
y = dataset.iloc[:,-1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()

label_l = [1,2,3]
for i in label_l:
    X[:, i] = labelencoder_X.fit_transform(X[:, i])
    
onehotencoder = OneHotEncoder(categorical_features = label_l)
X = onehotencoder.fit_transform(X).toarray()

# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Feature Scaling (scaling all attributes/featues in the same scale)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

## feature selection (rank all features in the range of 0,1)
#
#from sklearn.cross_validation import cross_val_score, ShuffleSplit
#from sklearn.ensemble import RandomForestRegressor
#rf = RandomForestRegressor(n_estimators=15, criterion = 'mse', n_jobs=1)
#
#scores = []
#for i in range(len(X_columns)):
#    score = cross_val_score(rf, X[:, i:i+1], y, scoring="r2",cv=ShuffleSplit(len(X), 3, .3))
#    scores.append((round(np.mean(score), 3), X_columns[i]))
#
#data = sorted(scores, reverse=True)
#df_result = pd.DataFrame(data,columns=['rank','feature'])
#
##sample code for feature scalling: feature ranking result is scalled in the range of  0 to 1
#def scale_a_number(inpt, to_min, to_max, from_min, from_max):
#    return (to_max-to_min)*(inpt-from_min)/(from_max-from_min)+to_min
#
#def scale_a_list(l, to_min, to_max):
#    return [scale_a_number(i, to_min, to_max, min(l), max(l)) for i in l]
#
##scale rank in to 0,1 range
#df_result.insert(1,"scaled_rank",scale_a_list(df_result['rank'], 0, 1))


# excluding few features those has none or little effect on classification result.
# This will simplyfy the model (less overfitting)
#X= X[:, :-2]

## apply different classifer from below, uncomment the one you like to be in action
from sklearn.model_selection import KFold, cross_val_score
classifier=''
y_pred_all=np.empty(shape=(0)) #empty 1d numpy array
proba_all=np.empty(shape=(0,2)) # empty 2d numpy array-> o rows 2 column

## SVM
#from sklearn.svm import SVC
#classifier = SVC(kernel = 'linear', random_state = 0, probability=True)

## random forest
#from sklearn.ensemble import RandomForestClassifier
#classifier = RandomForestClassifier(n_estimators=10,criterion="gini")

##    Naive Bayes
#from sklearn.naive_bayes import BernoulliNB
#classifier = BernoulliNB() 

# gradient Boosting
#from sklearn.ensemble import GradientBoostingClassifier
#classifier = GradientBoostingClassifier()

# knn
#from sklearn.neighbors import KNeighborsClassifier
#classifier = KNeighborsClassifier()

# Extra Trees 
from sklearn.ensemble import ExtraTreesClassifier
classifier = ExtraTreesClassifier(criterion="entropy")

##we found extra Trees classifier was the best classifier with highest accuracy.To see other classifiers result 
#keep that two line of code uncommented 


# k fold cross validation
start = time.time()
k_fold = KFold(n_splits=10)

for train_indices, test_indices in k_fold.split(X):
    #print('Train: %s | test: %s' % (train_indices, test_indices))      
    X_train = X[train_indices[0]:train_indices[-1]+1]
    y_train = y[train_indices[0]:train_indices[-1]+1]

    X_test = X[test_indices[0]:test_indices[-1]+1]
    y_test = y[test_indices[0]:test_indices[-1]+1]

    # Fitting SVM to the Training set    
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    y_pred_all=np.concatenate((y_pred_all,y_pred),axis=0)

    proba = classifier.predict_proba(X_test)
    proba_all=np.concatenate((proba_all,proba))

end = time.time()
diff = end - start
print("classification time:")
print(diff)

# this gives us how strong is the ye/no decision with a probability value (continuous value 
# rather than just the discrete binary decision)
df_result = pd.DataFrame(proba_all,columns=['probability_no','probability_yes'])

#df_result.head()

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
precision, recall, fscore, support = precision_recall_fscore_support(y, y_pred_all,average='binary')

df_metrics = pd.DataFrame([[acsc, precision, recall, fscore]], 
                        index=[0],
                        columns=['accuracy','precision', 'recall', 'fscore'])

conn = sqlite3.connect("credit.sqlite")
curr = conn.cursor()


# store result in database
result_1 = pd.DataFrame(columns = ['account', 'offline', 'online', 'total', 'default_olap','default_actual','default'])
ac = 1
for index, row in df_result.iterrows():
    row_dict = {'account': int(ac), 'offline': row['probability_yes'],'online': 0,'total': row['probability_yes'],'default_olap': y_pred_all[index],'default_actual': y[index], 'default': y_pred_all[index]}
    result_1 = result_1.append(row_dict,ignore_index=True)
    ac = ac +1

result_1 = pd.concat([result_1,dataset_slice], axis =1)



cm_1 = pd.DataFrame(columns = ['TruePositives', 'TrueNegatives', 'FalsePositives', 'FalseNegatives'])
for index, row in df_cm.iterrows():
    row_dict = {'TruePositives': row['True Positives'], 'TrueNegatives': row['True Negatives'], 'FalsePositives': row['False Positives'],'FalseNegatives': row['False Negatives']}
    cm_1 = cm_1.append(row_dict,ignore_index=True)
cm_tb_name = "cm_" + str(serial)
cm_1.to_sql(cm_tb_name, conn, if_exists="replace")

metrics_1 = pd.DataFrame(columns = ['accuracy', 'precision', 'recall', 'fscore'])
for index, row in df_metrics.iterrows():
    row_dict = {'accuracy': row['accuracy'], 'precision': row['precision'], 'recall': row['recall'],'fscore': row['fscore']}
    metrics_1 = metrics_1.append(row_dict,ignore_index=True)
metrics_tb_name = "metrics_" + str(serial)
metrics_1.to_sql(metrics_tb_name, conn, if_exists="replace")


#update default field: if an  account is already default from OLTP run keep update that as default.


if serial > 1:
    #
    table_name = "result_" + str(serial-1)
    df_result = pd.read_sql_query("select * from " + table_name + ";",conn)
    
    for index, row in result_1.iterrows():
        #print(result_1.at[index, 'default'])
        #print(df_result.at[index, 'default'])
        if result_1.at[index, 'default'] <= df_result.at[index, 'default']:
            result_1.at[index, 'default'] = df_result.at[index, 'default']




table_name = "result_" + str(serial)
result_1.to_sql(table_name, conn, if_exists="replace")
index_name = "idx_res" + str(serial)
curr.execute("create index " + index_name+ " on " + table_name+ "(account);")

conn.commit()
curr.close()
conn.close()