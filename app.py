# -*- coding: utf-8 -*-
"""Created on Wed July 28 20:55:27 2017
@author: Sheikh Rabiul Islam
Purpose: Research work on mining bad credit account from both OLAP and OLTP system data.
"""
#import classes,API's
from flask import Flask,render_template,request,json,send_from_directory,send_file, make_response
from werkzeug import secure_filename
import os
import time
import datetime
import sqlite3
import pandas as pd
import numpy as np
import sys,csv
app = Flask(__name__)

#global config variables
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['csv'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 #maximum allowed file 16 Mb

#route to home of the site
@app.route('/')
def main():
    return render_template('index.html')  

#route to Data Selection   
@app.route('/data_selection')
def data_selection():
    return render_template('data_selection.html')  

#route to Feature Selection
@app.route('/feature_selection')
def feature_selection():
    return render_template('feature_selection.html')

#route to algorithm selection
@app.route('/algorithm_selection')
def algorithm_selection():
    return render_template('algorithm_selection.html')

#route to OLAP Data Selection    
@app.route('/data_selection_oltp')
def data_selection_oltp():
    return render_template('data_selection_oltp.html')

#route to Build Model section
@app.route('/build_model')
def build_model():
    return render_template('build_model.html')

#route to Run Model section
@app.route('/run_model')
def run_model():
    return render_template('run_model.html')

#route to  Result Evaluation section
@app.route('/result_evaluation')
def result_evaluation():
    return render_template('result_evaluation.html')

#sample code to send file from directory
@app.route('/show/')
def show():
    return send_from_directory(app.config['UPLOAD_FOLDER'], 'data_selection.csv')

#let user download the file
@app.route('/downloads/<filename>')
def downloads(filename):
    outdata=''
    with open(os.path.join(app.config['UPLOAD_FOLDER'], filename)) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            outdata += ",".join(row) + "\n"
    
    response = make_response(outdata)
    response.headers["Content-Disposition"] = "attachment; filename=data.csv"
    return response

#sample code to send file
@app.route('/show/static-pdf/')
def show_static_pdf():
    with open('uploads/data_selection.csv', 'rb') as static_file:
        return send_file(static_file, attachment_filename='data_selection.csv')

#sample code for feature scalling: feature ranking result is scalled in the range of  0 to 1
def scale_a_number(inpt, to_min, to_max, from_min, from_max):
    return (to_max-to_min)*(inpt-from_min)/(from_max-from_min)+to_min

def scale_a_list(l, to_min, to_max):
    return [scale_a_number(i, to_min, to_max, min(l), max(l)) for i in l]

#Run selected algorithm on the OLAP data to get offline risk probability
@app.route('/algorithm_selection_run', methods = ['GET', 'POST'])
def algorithm_selection_run():
    msg=''  
    data = list()
    df_result = pd.DataFrame()
    if request.method == 'POST':
      from_url = request.form.get('from_url')
      algorithm = request.form.get('algorithm')
      #time.sleep(1)
      msg = ''
      
      conn = sqlite3.connect("databases/credit.sqlite")
      curr = conn.cursor()
      dataset = pd.read_sql_query("select * from olap;",conn)
      X = dataset.iloc[:,2:-6].values
      X_columns = dataset.iloc[:,2:-6].columns.values
      y = dataset.iloc[:,-1].values
            
      # Encoding categorical data -> coded caregorical features into numerica/intiger features
      # Encoding the Independent Variable
      from sklearn.preprocessing import LabelEncoder, OneHotEncoder
      labelencoder_X = LabelEncoder()
      #index    account    status_of_existing_checking_account    duration    credit_history    purpose    credit_amount     saving_account_or_bonds    pesent_employment_since    
      #  0       1          2                                       3         4             5             6           7                               8                     
      #installment_rate    personal_status_and_sex    other_debtor_or_guarantors     present_residence_since    property    age    other_installment_plans    
      #   9                       10                          11                      12                     13    14       15
      #housing     number_of_existing_credit_this_bank    job    people_for_maintenance    telephone    foreign_worker air_ticket_purchase   out_of_the_country    payment_due_date
      # 16           17                              18     19                          20            21                  22                  23                        24    
      #mimum_due_paid    location    avg_number_of_transaction   class
      #        25        26     27          28                     29
      #label_l = [0,2,3,5,6,8,9,11,13,14,16,18,19]
      #label_l_all = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,21,22]
      #label_l = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
      label_l = [0,2,3,5,6,8,9,11,13,14,16,18,19]
      
      for i in label_l:
          X[:, i] = labelencoder_X.fit_transform(X[:, i])
    
      # Transforms each categorical feature with m posssible values into m binary features, with only one active perrecords. This helps treating values are not ordered. 
      onehotencoder = OneHotEncoder(categorical_features = label_l)
      X = onehotencoder.fit_transform(X).toarray()
      
      # Encoding the Dependent Variable -> coded caregorical features into numerica/intiger features
      labelencoder_y = LabelEncoder()
      y = labelencoder_y.fit_transform(y)
    
      # Feature Scaling -> scale all features in the same maximum and minimum ranges to avoid biases. 
      from sklearn.preprocessing import StandardScaler
      sc = StandardScaler()
      X = sc.fit_transform(X)      
            
      
      from sklearn.model_selection import KFold, cross_val_score
      classifier=''
      y_pred_all=np.empty(shape=(0)) #empty 1d numpy array
      proba_all=np.empty(shape=(0,2)) # empty 2d numpy array-> o rows 2 column
      
      if algorithm == 'svm':
          # support vector machine
          from sklearn.svm import SVC
          classifier = SVC(kernel = 'linear', random_state = 0, probability=True)
      elif algorithm == 'rf':
          # random forest
          from sklearn.ensemble import RandomForestClassifier
          classifier = RandomForestClassifier(n_estimators=10,criterion="gini")
      elif algorithm == 'bnb':
          #Bernoulli  Naive bayes
          from sklearn.naive_bayes import BernoulliNB
          classifier = BernoulliNB() 
      elif algorithm =='gb':
          from sklearn.ensemble import GradientBoostingClassifier
          classifier = GradientBoostingClassifier()
      elif algorithm =='knn':
          from sklearn.neighbors import KNeighborsClassifier
          classifier = KNeighborsClassifier()
      elif algorithm =='et':
          from sklearn.ensemble import ExtraTreesClassifier
          classifier = ExtraTreesClassifier(criterion="entropy")
          
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
        
      df_result = pd.DataFrame(proba_all,columns=['probability_good','probability_bad'])
      df_result_db = pd.DataFrame(proba_all[:,1],columns=['offline'])
      df_result_db.sort_index(inplace=True)

      df_result.insert(0,'account',dataset['account'])
      df_result_db.insert(0,'account',dataset['account'])
     
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
      precision, recall, fscore, support = precision_recall_fscore_support(y, y_pred_all,average='weighted')
      
      df_metrics = pd.DataFrame([[acsc, precision, recall, fscore]], 
                                index=[0],
                                columns=['accuracy','precision', 'recall', 'fscore'])
      
      #insert result in to the database
      conn = sqlite3.connect("databases/credit.sqlite")
      curr = conn.cursor()
      curr.execute("delete from result_olap;")
      curr.execute("update sqlite_sequence set seq=0 where name='result_olap';")
      df_result_db.to_sql("result_olap", conn, if_exists="append")
      conn.commit()
      curr.close()
      conn.close()
    return render_template(from_url,tables=[df_metrics.to_html(),df_cm.to_html(),df_result.to_html()],titles=['na','Brief Result','Confusion Matrix','Detail Result'])


#feature selection algorithms
@app.route('/feature_selection_run', methods = ['GET', 'POST'])
def feature_selection_run():
    msg=''  
    data = list()
    df_result = pd.DataFrame()
    if request.method == 'POST':
      from_url = request.form.get('from_url')
      algorithm = request.form.get('algorithm')
      #time.sleep(1)
      msg = ''
      # Importing the dataset
      #######read from file
      conn = sqlite3.connect("databases/credit.sqlite")
      curr = conn.cursor()
      dataset = pd.read_sql_query("select * from olap;",conn)
      X = dataset.iloc[:,2:-6].values
      X_columns = dataset.iloc[:,2:-6].columns.values
      y = dataset.iloc[:,-1].values
      # Encoding categorical data
      # Encoding the Independent Variable
      from sklearn.preprocessing import LabelEncoder, OneHotEncoder
      labelencoder_X = LabelEncoder()
      #index    account    status_of_existing_checking_account    duration    credit_history    purpose    credit_amount     saving_account_or_bonds    pesent_employment_since    
      #  0       1          2                                       3         4             5             6           7                               8                     
      #installment_rate    personal_status_and_sex    other_debtor_or_guarantors     present_residence_since    property    age    other_installment_plans    
      #   9                       10                          11                      12                     13    14       15
      #housing     number_of_existing_credit_this_bank    job    people_for_maintenance    telephone    foreign_worker air_ticket_purchase   out_of_the_country    payment_due_date
      # 16           17                              18     19                          20            21                  22                  23                        24    
      #mimum_due_paid    location   avg_number_of_transaction  class
      #        25        26     27       28                       29
      #label_l = [0,2,3,5,6,8,9,11,13,14,16,18,19]
      #label_l_all = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,21,22]
      #label_l = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
      label_l = [0,2,3,5,6,8,9,11,13,14,16,18,19]
      
      for i in label_l:
          X[:, i] = labelencoder_X.fit_transform(X[:, i])
    
      onehotencoder = OneHotEncoder(categorical_features = label_l)
      X = onehotencoder.fit_transform(X).toarray()
      # Encoding the Dependent Variable
      labelencoder_y = LabelEncoder()
      y = labelencoder_y.fit_transform(y)
    
      # Feature Scaling
      from sklearn.preprocessing import StandardScaler
      sc = StandardScaler()
      X = sc.fit_transform(X)      
      
      if algorithm == 'ft':
          #
          # Build a forest and compute the feature importances
          from sklearn.cross_validation import cross_val_score, ShuffleSplit
          from sklearn.ensemble import RandomForestRegressor
          forest = RandomForestRegressor()
          forest.fit(X, y)
          importances = forest.feature_importances_
          std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
        
          data= (sorted(zip(map(lambda x: round(x, 4), forest.feature_importances_), X_columns),reverse=True))
          #converting the list to a datafreame
          df_result = pd.DataFrame(data,columns=['rank','feature'])
          
          msg = 'Feature selected and ranked successfully using algorithm Forest of Trees .'
      elif algorithm == 'rf':
        from sklearn.cross_validation import cross_val_score, ShuffleSplit
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=15, criterion = 'mse', n_jobs=1)
        scores = []
        print(X_columns.shape)
        for i in range(len(X_columns)):
          score = cross_val_score(rf, X[:, i:i+1], y, scoring="r2",cv=ShuffleSplit(len(X), 3, .3))
          scores.append((round(np.mean(score), 3), X_columns[i]))
  
        data = sorted(scores, reverse=True)
        df_result = pd.DataFrame(data,columns=['rank','feature'])
        msg = 'Feature selected and ranked successfully using algorithm Random Forest.'
        
      elif algorithm == 'rfmda':
          #
          from sklearn.ensemble import RandomForestRegressor
          from sklearn.cross_validation import ShuffleSplit
          from sklearn.metrics import r2_score
          from collections import defaultdict
          rf = RandomForestRegressor()
          scores = defaultdict(list)     
          
          #crossvalidate the scores on a number of different random splits of the data
          for train_idx, test_idx in ShuffleSplit(len(X), 100, .3):
              X_train, X_test = X[train_idx], X[test_idx]
              y_train, y_test = y[train_idx], y[test_idx]
              r = rf.fit(X_train, y_train)
              acc = r2_score(y_test, rf.predict(X_test))
              for i in range(len(X_columns)):
                  X_t = X_test.copy()
                  np.random.shuffle(X_t[:, i])
                  shuff_acc = r2_score(y_test, rf.predict(X_t))
                  scores[X_columns[i]].append((acc-shuff_acc)/acc)
                  
          data= sorted([(round(np.mean(score), 4), feat) for feat, score in scores.items()], reverse=True)
          df_result = pd.DataFrame(data,columns=['rank','feature'])
          
          msg = 'Feature selected and ranked successfully using algorithm Random Forest (with mean decrease accuracy).'
     
      #df_result['rank_scaled']=scale_a_list(df_result['rank'], 0, 1)
      df_result.insert(1,"scaled_rank",scale_a_list(df_result['rank'], 0, 1))
      
      #insert feature ranking in to the database
      conn = sqlite3.connect("databases/credit.sqlite")
      curr = conn.cursor()
      curr.execute("delete from feature_ranking;")
      curr.execute("update sqlite_sequence set seq=0 where name='feature_ranking';")
      df_result.to_sql("feature_ranking", conn, if_exists="append")
      conn.commit()
      curr.close()
      conn.close()
      
      #return df
    return render_template(from_url,tables=[df_result.to_html()],titles=msg)

#upload file 
@app.route('/upload_file', methods = ['GET', 'POST'])
def upload_file():
    msg=''
    if request.method == 'POST':
      f = request.files['file']
      from_url = request.form.get('from_url')
      if f.filename == '':
            msg="Please select a file to proceed"
      else:
          filename = secure_filename(f.filename)
          f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
          msg = "File uploaded successfully ....."

      if from_url == 'data_selection.html':
          df=pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))
          conn = sqlite3.connect("databases/credit.sqlite")
          curr = conn.cursor()
          
          # inserting only new records keeping records for existing accounts
          olap_data = pd.read_sql_query("select * from olap;",conn)
          olap_data.pop("index")
          df_final = olap_data.append(df, ignore_index=True)
          df_final2 = df_final.drop_duplicates(subset = ['account'], keep = 'first')
          curr.execute("delete from olap;")
          curr.execute("update sqlite_sequence set seq=0 where name='olap';")
          df_final2.to_sql("olap", conn, if_exists="append")
          conn.commit()
          curr.close()
          conn.close()
          
      elif from_url == 'data_selection_oltp.html':
          df=pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))
          conn = sqlite3.connect("databases/credit.sqlite")
          curr = conn.cursor()
          curr.execute("delete from oltp;")
          curr.execute("update sqlite_sequence set seq=0 where name='oltp';")
          df.to_sql("oltp", conn, if_exists="append")
          conn.commit()
          curr.close()
          conn.close()
          
    return render_template(from_url,message=msg)
  
#upload file --default
@app.route('/upload_file_default', methods = ['GET', 'POST'])
def upload_file_default():
    msg=''
    if request.method == 'POST':
      from_url = request.form.get('from_url')
      msg = ''
      if from_url == 'data_selection.html':
          filename = 'dataset_olap_default.csv'
          df=pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))
          conn = sqlite3.connect("databases/credit.sqlite")
          curr = conn.cursor()
          curr.execute("delete from olap;")
          curr.execute("update sqlite_sequence set seq=0 where name='olap';")
          df.to_sql("olap", conn, if_exists="append")
          msg = "File selected successfully."
          conn.commit()
          curr.close()
          conn.close()
          
      elif from_url == 'data_selection_oltp.html':
          filename = 'dataset_oltp_default.csv'
          df=pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))
          conn = sqlite3.connect("databases/credit.sqlite")
          curr = conn.cursor()
          curr.execute("delete from oltp;")
          curr.execute("update sqlite_sequence set seq=0 where name='oltp';")
          df.to_sql("oltp", conn, if_exists="append")
          msg = "File selected successfully."
          conn.commit()
          curr.close()
          conn.close()

    return render_template(from_url,message=msg)

#shows stored data from a table
@app.route('/view_data/<table_name>')    
@app.route('/view_data/<table_name>/<order_by>')
def view_data(table_name,order_by=''):
    if order_by=='':
        order_by='index'
    sql=''
    sql = "select * from "+ table_name +" order by " + order_by + " limit 5000;"
    if order_by == 'index':
        sql = "select * from "+ table_name +" order by `index` limit 5000;"
    
    conn = sqlite3.connect("databases/credit.sqlite",timeout=10)
    curr = conn.cursor()
    data = pd.read_sql_query(sql,conn) 
    curr.close()
    conn.close()
    return render_template('view_data.html',tables=[data.to_html()],titles=table_name)
   
#clear screen -> make 5 blank line inbetween last andrecent output of the console    
def cls(): print ("\n" * 5)

#make model based on available configurations and rule setting
@app.route('/build_model_action', methods = ['GET', 'POST'])
def build_model_action():
    conn = sqlite3.connect("databases/credit.sqlite",timeout=10)
    curr = conn.cursor()
    sql="select a.`adt_rules_id`, max(b.`scaled_rank`) as 'impact_coefficient' from `adt_rules_feature_ranking_mapping` a,\
    `feature_ranking` b where a.`feature_ranking_id` = b.`index` group by  a.`adt_rules_id` order by a.`adt_rules_id`;"
    rules_mapping = pd.read_sql_query(sql,conn)
    
    for index, row in rules_mapping.iterrows():
        adt_id = row['adt_rules_id']
        ic = row['impact_coefficient']
        sql2="update adt_rules set impact_coefficient=? where id=?;"
        curr.execute(sql2, (ic, adt_id))
        
    conn.commit()
    curr.close()
    conn.close()
    return render_template('build_model.html', message="Model build successfuly.Please click next below.")


# Run the model and generate results
@app.route('/run_model_action', methods = ['GET', 'POST'])
def run_model_action():
    #db connection
    conn = sqlite3.connect("databases/credit.sqlite",timeout=10)
    curr = conn.cursor()
    
    #load tables from database in to memory [dataframe]
    oltp_data = pd.read_sql_query("select * from oltp;",conn)
    olap_data = pd.read_sql_query("select * from olap;",conn)
    config = pd.read_sql_query("select * from config;",conn)
    result = pd.read_sql_query("select * from result_olap;",conn)
    result_final = pd.DataFrame(columns=('account', 'offline', 'online','event','cause','total','class'))
    
    # load configs
    avg_tran_plus_std = config.loc[0,'avg_tran_plus_std']
    avg_number_of_tran_plus_std = config.loc[0,'avg_number_of_tran_plus_std']
    bad_account_threshold = config.loc[0,'bad_account_threshold']
    lmbda = config.loc[0,'lambda']
    
    #clear make blank line on console to distinguise between last and current outpus
    cls()
    
    #for each oltp transaction of a batch 
    for i in range(len(oltp_data)):
        tran_type = oltp_data.loc[i,'transaction_type']
        account = oltp_data.loc[i,'account']
        amount = oltp_data.loc[i,'amount']
        transaction_location = oltp_data.loc[i,'location']
        transaction_time= oltp_data.loc[i,'date']
        
        #select corresponding olap data for this account
        olap_data_row = olap_data.iloc[np.where(olap_data.account.values == account)]
        olap_data_row_dicts = olap_data_row.to_dict(orient='records')
        olap_data_row_dict = olap_data_row_dicts[0]
        present_residence_since = olap_data_row_dict['present_residence_since']
        present_job_since = olap_data_row_dict['present_employment_since']
        out_of_the_country = olap_data_row_dict['out_of_the_country']
        air_ticket_purchase = olap_data_row_dict['air_ticket_purchase']
        payment_due_date = olap_data_row_dict['payment_due_date']
        due_amount = olap_data_row_dict['credit_amount']
        minimum_due_paid = olap_data_row_dict['minimum_due_paid']
        location = olap_data_row_dict['location']
        avg_number_of_transaction = olap_data_row_dict['avg_number_of_transaction']
        
        coefficient_sum = 0
        coefficient_valid_sum = 0
        events = list()
        causes = list()
        
        std_rules = pd.read_sql_query("select * from std_rules where applicable_to_tran_type =:param;",conn, params={'param':tran_type})
        
        #for each std rules of that transaction type
        for j in range(len(std_rules)):
            std_id = std_rules.loc[j,'id']
            std_rule = std_rules.loc[j,'rule']
            events.append(std_rule)
            # need to check whether the transaction follow the std rule
            std_fail=0 
            if tran_type == 'pay':
                if std_id == 1:
                    if amount > avg_tran_plus_std:
                        std_fail=1
                elif std_id == 3:
                    transaction_date_n= time.strptime(transaction_time, "%Y-%m-%d")
                    now=str(datetime.datetime.now())
                    yr_mn=  now[0:7]
                    payment_due_date_n =  (yr_mn+str(payment_due_date[-3:]))
                    payment_due_date_nn= time.strptime(payment_due_date_n, "%Y-%m-%d")
                    
                    if transaction_date_n > payment_due_date_nn:
                        std_fail=1
                elif std_id == 4:
                    if amount < minimum_due_paid:
                        std_fail = 1
                elif std_id == 5:
                    if amount < due_amount:
                        std_fail=1
            elif tran_type == 'exp':
                if std_id == 2:
                    if avg_number_of_transaction > avg_number_of_tran_plus_std:
                        std_fail=1
                    elif transaction_location != location:
                        std_fail=1
                
            #if any of the std rule is not followded then dig down to find a cause [adaptive rule]            
            if std_fail==1:     
                std_rule = std_rules.loc[j,'rule']
                events.append(std_rule)
                adt_rules = pd.read_sql_query("select a.* from adt_rules a, std_adt_mapping b  where b.std_id=std_id  and a.id=b.adt_id group by a.id;",conn)
                
                #for each relevant adaptive rules  check fulfill or not fulfill
                for k in range(len(adt_rules)):
                    #make a separate function to check adt rules with switch statement
                    adt_id = adt_rules.loc[k,'id']
                    adt_rule = adt_rules.loc[k,'rule']
                    adt_impact_coefficient = adt_rules.loc[k,'impact_coefficient']
                    coefficient_sum += adt_impact_coefficient
                    
                    if adt_id==1:
                        #Address  change
                        if present_residence_since<1:
                            coefficient_valid_sum += adt_impact_coefficient
                            causes.append(adt_rule)                     
                    elif adt_id==2:
                        #Airticket purchase
                        if air_ticket_purchase == 1:
                            coefficient_valid_sum += adt_impact_coefficient
                            causes.append(adt_rule)                     
                    elif adt_id==3:
                        #Job switch
                        if present_job_since == "A71":
                            #now unemployed
                            coefficient_valid_sum += adt_impact_coefficient
                            causes.append(adt_rule)              
                    elif adt_id==4:
                        #out of the country
                        if out_of_the_country == 1:
                            coefficient_valid_sum += adt_impact_coefficient
                            causes.append(adt_rule) 
        #calculate online risk probability
        risk_probability_online=0 
        if coefficient_sum > 0:
            risk_probability_online = (1 - (coefficient_valid_sum/coefficient_sum))*100
            risk_probability_online=risk_probability_online*lmbda
        
        #grab offline risk probability
        offline_s = result.loc[result['account'] == account,'offline'].values
        offline = offline_s[0]
        offline = (1-lmbda)*(offline*100)

        # total risk probability  
        events_str = '\n'.join(events) #concatenating elements of list, separating by newline
        causes_str = '\n'.join(causes)
        total = offline + risk_probability_online
        
        #class, 1=bad account, 0 = good account
        clss=0
        if total >= bad_account_threshold:
            clss =1
        
        #set current total riks probability of an account as the offline risk probability for the same account for next transaction
        row_dict = {'account': account, 'offline': offline, 'online': risk_probability_online, 'event':events_str, 'cause': causes_str,'total':total,'class':clss}
        result_final=result_final.append(row_dict, ignore_index=True)
        result.loc[result['account'] == account,'offline'] = total*.01
    
    #reset sequence
    curr.execute("delete from result;")
    curr.execute("update sqlite_sequence set seq=0 where name='result';")
    result_final.to_sql("result",conn,if_exists="append")
    
    #export final result into the database
    result.pop('index')
    curr.execute("delete from result_olap;")
    curr.execute("update sqlite_sequence set seq=0 where name='result_olap';")
    result.to_sql("result_olap",conn,if_exists="append")
    
    #commit and close db connection
    conn.commit()
    curr.close()
    conn.close()
    return render_template('run_model.html',tables=[result_final.to_html()],titles=['na','Result (1=Bad Account, 0 = Good Account)'])


@app.route('/result_evaluation_action', methods = ['GET', 'POST'])
def result_evaluation_action():
    conn = sqlite3.connect("databases/credit.sqlite",timeout=10)
    curr = conn.cursor()
    result = pd.read_sql_query("select * from `result`;",conn)
    
    result.loc[result['class'] == 1,'class'] = "Bad Account"
    result.loc[result['class'] == 0,'class'] = "Good Account"
    result_cp = result.copy()
    result.pop('index')
    result.pop('class')
    result.pop('event')
    result.pop('cause')
    
    config = pd.read_sql_query("select * from config;",conn)
    bad_account_threshold = config.loc[0,'bad_account_threshold']
    
    Data = []
    
    result_subset = result.head(n=25)
    for index, row in result_subset.iterrows():
        r={ "Date": row['account'], "Categories": [{ "Name": "Offline", "Value": row['offline'] }, { "Name": "Online", "Value": row['online'] }, { "Name": "Total", "Value": row['total'] }], "LineCategory": [{ "Name": "Risk Threshold", "Value": bad_account_threshold }] }
        Data.append(dict(r))
        
    #commit and close db connection
    conn.commit()
    curr.close()
    conn.close()
    return render_template('result_evaluation.html',data=json.dumps(Data),tables=[result_cp.to_html()],titles=['na','Detail Result'])

# allow any host, allow debug.
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=1)






