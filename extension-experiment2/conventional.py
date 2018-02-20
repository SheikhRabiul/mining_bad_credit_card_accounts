# Author: Sheikh Rabiul Islam
# Date: 01/07/2018
# Purpose: Predicting credit card clients with default on payment

#import modules
import pandas as pd   
import numpy as np
import time


# import data
dataset = pd.read_csv('default_of_credit_card_clients.csv', sep=',')

# seperate the dependent (target) variaable
X = dataset.iloc[:,1:-1].values
X_columns = dataset.iloc[:,1:-1].columns.values
y = dataset.iloc[:,-1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()

label_l = [2,3,4]
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

# excluding few features those has none or little effect on classification result.
# This will simplyfy the model (less overfitting)
#X= X[:, :-2]


## apply different classifer from below, uncomment the one you like to be in action
from sklearn.model_selection import KFold, cross_val_score
classifier=''
y_pred_all=np.empty(shape=(0)) #empty 1d numpy array
proba_all=np.empty(shape=(0,2)) # empty 2d numpy array-> o rows 2 column

## SVM
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0, probability=True)

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
#from sklearn.ensemble import ExtraTreesClassifier
#classifier = ExtraTreesClassifier(criterion="entropy")

##we found extra Trees classifier was the best classifier with highest accuracy.To see other classifiers result 
#keep that two line of code uncommented 


# k fold cross validation

k_fold = KFold(n_splits=10)
start = time.time()
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


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score
from sklearn.metrics import precision_recall_fscore_support
cm = confusion_matrix(y, y_pred_all)

#accuracy -number of instance correctly classified
acsc = accuracy_score(y, y_pred_all) 
df_cm = pd.DataFrame([[cm[1][1], cm[0][0],cm[0][1], cm[1][0]]], 
                        index=[0],
                        columns=['True Positives','True Negatives', 'False Positives', 'False Negatives'])
print(df_cm)
#precision, recall, fscore, support
precision, recall, fscore, support = precision_recall_fscore_support(y, y_pred_all,average='binary')

df_metrics = pd.DataFrame([[acsc, precision, recall, fscore]], 
                        index=[0],
                        columns=['accuracy','precision', 'recall', 'fscore'])

print(df_metrics)
