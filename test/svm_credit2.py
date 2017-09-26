# Support Vector Machine (SVM)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('dataset_olap_mini.csv')
#print(dataset.head())
X = dataset.iloc[:,1:-1].values
print(X)
y = dataset.iloc[:,-1].values

print(y)
#print(X)
#    0                                   10                  15                     22
#X=A11,6,A34,A43,1169,A65,A75,4,A93,A101,4,A121,67,A143,A152,2,A173,1,A192,A201,1,1

# Taking care of missing data
"""from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
"""
# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
X[:, 5] = labelencoder_X.fit_transform(X[:, 5])
X[:, 6] = labelencoder_X.fit_transform(X[:, 6])
X[:, 8] = labelencoder_X.fit_transform(X[:, 8])
X[:, 9] = labelencoder_X.fit_transform(X[:, 9])
X[:, 11] = labelencoder_X.fit_transform(X[:, 11])
X[:, 13] = labelencoder_X.fit_transform(X[:, 13])
X[:, 14] = labelencoder_X.fit_transform(X[:, 14])
X[:, 16] = labelencoder_X.fit_transform(X[:, 16])
X[:, 18] = labelencoder_X.fit_transform(X[:, 18])
X[:, 19] = labelencoder_X.fit_transform(X[:, 19])

onehotencoder = OneHotEncoder(categorical_features = [0,2,3,5,6,8,9,11,13,14,16,18,19])
X = onehotencoder.fit_transform(X).toarray()
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
#print(y)
# Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)



y_pred_all=np.empty(shape=(0)) #empty 1d numpy array
proba_all=np.empty(shape=(0,2)) # empty 2d numpy array-> o rows 2 column

from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0, probability=True)

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
    #print(proba)


print("y_pred_all")
print(y_pred_all)


print("proba_all")
print(proba_all)
df_result = pd.DataFrame(proba_all,columns=['probability(good)','probability(bad)'])
print(df_result)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score
from sklearn.metrics import precision_recall_fscore_support
cm = confusion_matrix(y, y_pred_all)

print(cm)
print(cm[1][1])

#accuracy -number of instance correctly classified
acsc = accuracy_score(y, y_pred_all)
print("Accuracy:")
print(acsc)

#precision, recall, fscore, support
precision, recall, fscore, support = precision_recall_fscore_support(y, y_pred_all,average='weighted')

df_metrics = pd.DataFrame([[acsc, precision, recall, fscore]], 
                          index=[0],
                          columns=['accuracy','precision', 'recall', 'fscore'])
print(df_metrics)

print("precision:")
print(precision)

print("recall:")
print(recall)

print("fscore:")
print(fscore)



"""
# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()"""