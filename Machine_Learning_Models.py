#importing libraries
import pandas as pd
import numpy as np
pd.set_option('display.max_column',500)
pd.set_option('display.width',1000)

#Visualization (get_ipython().run_line_magic('matplotlib', 'inline') ''OR'' %matplotlib inline )
import matplotlib.pyplot as plt
import seaborn as sns

#scaling 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

#train test split
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

#Logistic regression (classification)
from sklearn.linear_model import LogisticRegression

#backward and forward selection 
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

#exhaustive search
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS

#cross validation
from sklearn.model_selection import cross_val_score

#feature selection
from sklearn.feature_selection import SelectKBest, f_regression

#model evaluation
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score, f1_score)

#Random Forest
from sklearn.ensemble import RandomForestClassifier

#Decsion Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

#KNN model - K nearest neighbors
from sklearn.neighbors import KNeighborsClassifier

#hyperparameter tuning 
from sklearn.pipeline import Pipeline  
from sklearn.model_selection import GridSearchCV


#reading the merged file 
df = pd.read_csv("/Users/pallavijain/Desktop/data_6212/merged_df.csv")

df.info()
# 125 rows and 44 columns with no null values 


# # X and Y variables + Train and Test split

#defining x and y
x = df[["age","gender_female","race_asian","race_black","race_hispanic","race_north american native",
        "race_white","race_other","prescription_procurement_duration","active_status_1","claim_type_DME",
        "claim_type_I","claim_type_P","charge_amount","paid_amount","claim_duration","encounter_duration",
        "hcpcs_modifier_4_PO","hcpcs_modifier_4_GY","coverage_duration","medicare_payer_type",
        "primary_diagnosis_rank_1","request_status_active","dose_1","dose_NR","dose_ONE","dose_PRN",
       "dose_SCH","dose_STA","loinc_2708-6","loinc_8310-5","loinc_8462-4","loinc_8480-6","loinc_8867-4",
       "loinc_9279-1",]]
y=df["deceased_flag"]


x.shape


df['deceased_flag'].value_counts()

#12 people have deceased and 113 are alive in our data


#train test split 70-30

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=100)


# # Random Forest Classification Model on all 35 X variables 


##initialize and train
rfc=RandomForestClassifier(random_state=100,n_estimators=1000)
rfc.fit(x_train,y_train)



##getting predicted values for testing and training data
y_pred_test=rfc.predict(x_test)
y_pred_train=rfc.predict(x_train)



##model evaluation on testing data 
print("Accuracy score for testing data in RFC (before tuning) is", accuracy_score(y_test,y_pred_test)) #0.94
print("Precision score for testing data in RFC (before tuning) is", precision_score(y_test,y_pred_test))
print("Recall score for testing data in RFC (before tuning) is", recall_score(y_test,y_pred_test))
print("F1 score for testing data in RFC (before tuning) is", f1_score(y_test,y_pred_test))




##Confusion Matrix for testing data 
print("Confusion Matrix for testing data in RFC (before tuning) is")
pd.DataFrame(confusion_matrix(y_test,y_pred_test),index = ['Actual : 0', 'Actual : 1'],
             columns =['pred : 0', 'pred : 1'])


##model evaluation on training data - overfitting occured
print("Accuracy score for training data in RFC (before tuning) is", accuracy_score(y_train,y_pred_train)) 
print("Precision score for training data in RFC (before tuning) is", precision_score(y_train,y_pred_train))
print("Recall score for training data in RFC (before tuning) is", recall_score(y_train,y_pred_train))
print("F1 score for training data in RFC (before tuning) is", f1_score(y_train,y_pred_train))


##Confusion Matrix for training data  
print("Confusion Matrix for training data in RFC (before tuning) is")
pd.DataFrame(confusion_matrix(y_train,y_pred_train),index = ['Actual : 0', 'Actual : 1'],
             columns =['pred : 0', 'pred : 1'])


# # Tuned Random Forest Classfication Model (35 features)

##tuning the hyper parameters with K fold
parameter_grid={'n_estimators': range(100,200)}
cv = KFold(n_splits=10, random_state=1, shuffle=True)
grid=GridSearchCV(rfc,parameter_grid,verbose=3,scoring='f1',cv=10)



#seeing which n_estimators produce the best recall
scores =[]
for k in range(1, 200):
    rfc = RandomForestClassifier(n_estimators=k)
    rfc.fit(x_train, y_train)
    y_pred = rfc.predict(x_test)
    scores.append(recall_score(y_test, y_pred))

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# plot the relationship between K and testing recall
# plt.plot(x_axis, y_axis)
plt.plot(range(1, 200), scores)
plt.xlabel('Value of n_estimators for Random Forest Classifier')
plt.ylabel('Testing Recall')


grid.fit(x_train,y_train) 


##what are the best hyper parameters :: {'n_estimators': 100}
grid.best_params_

#initialize and training the model with tuned parameters
rfc=RandomForestClassifier(random_state=1,n_estimators=100)
rfc.fit(x_train,y_train)


##getting predicted values for testing and training data
y_pred_test=rfc.predict(x_test)
y_pred_train=rfc.predict(x_train)


##model evaluation on testing data 
print("Accuracy score for testing data in RFC (after tuning) is", accuracy_score(y_test,y_pred_test))
print("Precision score for testing data in RFC (after tuning) is", precision_score(y_test,y_pred_test))
print("Recall score for testing data in RFC (after tuning) is", recall_score(y_test,y_pred_test))
print("F1 score for testing data in RFC (after tuning) is", f1_score(y_test,y_pred_test))



##Confusion Matrix for testing data 
print("Confusion Matrix for testing data in RFC (after tuning) is")
pd.DataFrame(confusion_matrix(y_test,y_pred_test),index = ['Actual : 0', 'Actual : 1'],
             columns =['pred : 0', 'pred : 1'])

##model evaluation on training data - overfitting
print("Accuracy score for training data in RFC (after tuning) is", accuracy_score(y_train,y_pred_train))
print("Precision score for training data in RFC (after tuning) is", precision_score(y_train,y_pred_train))
print("Recall score for training data in RFC (after tuning) is", recall_score(y_train,y_pred_train))
print("F1 score for training data in RFC (after tuning) is", f1_score(y_train,y_pred_train))



##Confusion Matrix for training data  
print("Confusion Matrix for training data in RFC (after tuning) is")
pd.DataFrame(confusion_matrix(y_train,y_pred_train),index = ['Actual : 0', 'Actual : 1'],
             columns =['pred : 0', 'pred : 1'])


# # Decision Tree Classification Model on all 35 X variables


'''Decision Tree Classification Model before tuning'''
#initialize and train
dt=DecisionTreeClassifier(random_state=1)
dt.fit(x_train,y_train)



##getting predicted values for testing and training data
y_pred_test=dt.predict(x_test)
y_pred_train=dt.predict(x_train)


##model evaluation on testing data 
print("Accuracy score for testing data in DT (before tuning) is", accuracy_score(y_test,y_pred_test))
print("Precision score for testing data in DT (before tuning) is", precision_score(y_test,y_pred_test))
print("Recall score for testing data in DT (before tuning) is", recall_score(y_test,y_pred_test))
print("F1 score for testing data in DT (before tuning) is", f1_score(y_test,y_pred_test))



##Confusion Matrix for testing data  
print("Confusion Matrix for testing data in DT (before tuning) is")
pd.DataFrame(confusion_matrix(y_test,y_pred_test),index = ['Actual : 0', 'Actual : 1'],
             columns =['pred : 0', 'pred : 1'])


##model evaluation on training data - OVERFITTING 
print("Accuracy score for training data in DT (before tuning) is", accuracy_score(y_train,y_pred_train))
print("Precision score for training data in DT (before tuning) is", precision_score(y_train,y_pred_train))
print("Recall score for training data in DT (before tuning) is", recall_score(y_train,y_pred_train))
print("F1 score for training data in DT (before tuning) is", f1_score(y_train,y_pred_train))


##Confusion Matrix for training data  
print("Confusion Matrix for training data in DT (before tuning) is")
pd.DataFrame(confusion_matrix(y_train,y_pred_train),index = ['Actual : 0', 'Actual : 1'],
             columns =['pred : 0', 'pred : 1'])



#plotting the tree with classes
class_names = ['Survived', 'Deceased']
plt.figure(figsize=(15, 15))
plot_tree(dt, class_names=class_names)



#first split in untuned tree: x[6] race_white , followed by age x[0] and claim_duration x[15]
x.info()


# # Tuned Decision Tree Classification Model (35 features)


##tuning the hyper parameters with K-fold
parameter_grid={'max_depth': range(1,18), 'min_samples_split': range(2,40)}
cv = KFold(n_splits=10, random_state=1, shuffle=True)
grid=GridSearchCV(dt,parameter_grid,verbose=3,scoring='f1',cv=10)


grid.fit(x_train,y_train)



##seeing the best hyper parameters : {'max_depth': 2, 'min_samples_split': 2}
grid.best_params_


#initialize and training the model with tuned parameters
dt=DecisionTreeClassifier(max_depth=2,min_samples_split=2,random_state=1) 
dt.fit(x_train,y_train) 


##getting predicted values for testing and training data
y_pred_test=dt.predict(x_test)
y_pred_train=dt.predict(x_train)



##model evaluation on testing data 
print("Accuracy score for testing data in DT (after tuning) is", accuracy_score(y_test,y_pred_test))
print("Precision score for testing data in DT (after tuning) is", precision_score(y_test,y_pred_test))
print("Recall score for testing data in DT (after tuning) is", recall_score(y_test,y_pred_test))
print("F1 score for testing data in DT (after tuning) is", f1_score(y_test,y_pred_test))


##Confusion Matrix for testing data  
print("Confusion Matrix for testing data in DT (after tuning) is")
pd.DataFrame(confusion_matrix(y_test,y_pred_test),index = ['Actual : 0', 'Actual : 1'],
             columns =['pred : 0', 'pred : 1'])


##model evaluation on training data 
print("Accuracy score for training data in DT (after tuning) is", accuracy_score(y_train,y_pred_train))
print("Precision score for training data in DT (after tuning) is", precision_score(y_train,y_pred_train))
print("Recall score for training data in DT (after tuning) is", recall_score(y_train,y_pred_train))
print("F1 score for training data in DT (after tuning) is", f1_score(y_train,y_pred_train))



##Confusion Matrix for training data  
print("Confusion Matrix for training data in DT (after tuning) is")
pd.DataFrame(confusion_matrix(y_train,y_pred_train),index = ['Actual : 0', 'Actual : 1'],
             columns =['pred : 0', 'pred : 1'])



#plotting the tree with tuned parameters
class_names = ['Survived', 'Deceased']
plt.figure(figsize=(15, 15))
plot_tree(dt, class_names=class_names)



x.info()
#using the same first split as the tree before tuning: x[6] race_white
# the model takes coverage_duration x[15](if the race is white) and age x[0] (if the race is not white) for second split


# # Logistic Regression (35 X variables - scaled)

#scaling for Logit model
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled=scaler.transform(x_test)



logmodel = LogisticRegression(solver='liblinear') 
logmodel.fit(x_train_scaled,y_train)


#getting the logistic equation
logmodel.intercept_
logmodel.coef_


# get predicted probabilities between 0 and 1
y_pred_test=logmodel.predict(x_test_scaled)
y_pred_train=logmodel.predict(x_train_scaled)



##model evaluation on testing data 
print("Accuracy score for testing data in logit model is", accuracy_score(y_test,y_pred_test))
print("Precision score for testing data in logit model is", precision_score(y_test,y_pred_test))
print("Recall score for testing data in logit model is", recall_score(y_test,y_pred_test))
print("F1 score for testing data in logit model is", f1_score(y_test,y_pred_test))


##Confusion Matrix for testing data 
print("Confusion Matrix for testing data in logit model is")
pd.DataFrame(confusion_matrix(y_test,y_pred_test),index = ['Actual : 0', 'Actual : 1'],
             columns =['pred : 0', 'pred : 1'])



##model evaluation on training data 
print("Accuracy score for training data in logit model is", accuracy_score(y_train,y_pred_train))
print("Precision score for training data in logit model is", precision_score(y_train,y_pred_train))
print("Recall score for training data in logit model is", recall_score(y_train,y_pred_train))
print("F1 score for training data in logit model is", f1_score(y_train,y_pred_train))


#Confusion Matrix for training data  
print("Confusion Matrix for training data in logit model is")
pd.DataFrame(confusion_matrix(y_train,y_pred_train),index = ['Actual : 0', 'Actual : 1'],
             columns =['pred : 0', 'pred : 1'])


# # Forward Feature Selection for Logit Model


#initialize
lr = LogisticRegression(solver='liblinear') 
cv = KFold(n_splits=10, random_state=1, shuffle=True)
sfs = SFS(logmodel, 
          k_features=(1,35), 
          forward=True,
           scoring='f1',
         cv=10)



#train
sfs.fit(x_train, y_train)


#seeing which features were selected
sfs.k_feature_names_   



#only 8 out of 35 features
'''('age',
 'gender_female',
 'race_asian',
 'race_black',
 'race_hispanic',
 'race_north american native',
 'race_white',
 'coverage_duration')'''


#transformed data will have only selected features
X_train_sfs = sfs.transform(x_train)
X_test_sfs = sfs.transform(x_test)



#Fit the model using the new feature subset 
lr.fit(X_train_sfs, y_train)



###getting predicted values for testing and training data
y_pred_test = lr.predict(X_test_sfs) 
y_pred_train = lr.predict(X_train_sfs) 


#model evaluation on testing data 
print("Accuracy score for testing data in logit model (forward feature selection) is", accuracy_score(y_test,y_pred_test))
print("Precision score for testing data in logit model (forward feature selection) is", precision_score(y_test,y_pred_test))
print("Recall score for testing data in logit model (forward feature selection) is", recall_score(y_test,y_pred_test))
print("F1 score for testing data in logit model (forward feature selection) is", f1_score(y_test,y_pred_test))


#Confusion Matrix for testing data  
print("Confusion Matrix for testing data in logit model (forward feature selection) is")
pd.DataFrame(confusion_matrix(y_test,y_pred_test),index = ['Actual : 0', 'Actual : 1'],
             columns =['pred : 0', 'pred : 1'])


##model evaluation on training data 
print("Accuracy score for training data in logit model (forward feature selection) is", accuracy_score(y_train,y_pred_train))
print("Precision score for training data in logit model (forward feature selection) is", precision_score(y_train,y_pred_train))
print("Recall score for training data in logit model (forward feature selection) is", recall_score(y_train,y_pred_train))
print("F1 score for training data in logit model (forward feature selection) is", f1_score(y_train,y_pred_train))


#Confusion Matrix for training data  
print("Confusion Matrix for training data in logit model (forward feature selection) is")
pd.DataFrame(confusion_matrix(y_train,y_pred_train),index = ['Actual : 0', 'Actual : 1'],
             columns =['pred : 0', 'pred : 1'])


# # Backward Feature Selection for Logit Model


#initialize
lr = LogisticRegression(solver='liblinear') 
cv = KFold(n_splits=10, random_state=1, shuffle=True)
sfs = SFS(logmodel, 
          k_features=(1,35), 
          forward=False,
           scoring='f1',
         cv=10)

#train
sfs.fit(x_train, y_train)


#seeing which features were selected
sfs.k_feature_names_   



## 21 out of 35 features were selected
'''('age',
 'gender_female',
 'race_asian',
 'race_black',
 'race_hispanic',
 'race_north american native',
 'race_white',
 'race_other',
 'prescription_procurement_duration',
 'active_status_1',
 'claim_type_DME',
 'claim_type_I',
 'claim_type_P',
 'paid_amount',
 'claim_duration',
 'coverage_duration',
 'medicare_payer_type',
 'dose_1',
 'dose_PRN',
 'dose_SCH',
 'loinc_8310-5')'''

#transformed data will have only selected features
X_train_sfs = sfs.transform(x_train)
X_test_sfs = sfs.transform(x_test)



#Fit the model using the new feature subset 
lr.fit(X_train_sfs, y_train)



###getting predicted values for testing and training data
y_pred_test = lr.predict(X_test_sfs) 
y_pred_train = lr.predict(X_train_sfs) 


##model evaluation on testing data 
print("Accuracy score for testing data in logit model (backward feature selection) is", accuracy_score(y_test,y_pred_test))
print("Precision score for testing data in logit model (backward feature selection) is", precision_score(y_test,y_pred_test))
print("Recall score for testing data in logit model (backward feature selection) is", recall_score(y_test,y_pred_test))
print("F1 score for testing data in logit model (backward feature selection) is", f1_score(y_test,y_pred_test))


#Confusion Matrix for testing data  
print("Confusion Matrix for testing data in logit model (backward feature selection) is")
pd.DataFrame(confusion_matrix(y_test,y_pred_test),index = ['Actual : 0', 'Actual : 1'],
             columns =['pred : 0', 'pred : 1'])


##model evaluation on training data 
print("Accuracy score for training data in logit model (backward feature selection) is", accuracy_score(y_train,y_pred_train))
print("Precision score for training data in logit model (backward feature selection) is", precision_score(y_train,y_pred_train))
print("Recall score for training data in logit model (backward feature selection) is", recall_score(y_train,y_pred_train))
print("F1 score for training data in logit model (backward feature selection) is", f1_score(y_train,y_pred_train))



#Confusion Matrix for training data  
print("Confusion Matrix for training data in logit model (backward feature selection) is")
pd.DataFrame(confusion_matrix(y_train,y_pred_train),index = ['Actual : 0', 'Actual : 1'],
             columns =['pred : 0', 'pred : 1'])


# # EFS Logit Model


##training 
lr = LogisticRegression()
cv = KFold(n_splits=2, random_state=1, shuffle=True)
efs = EFS(lr, 
          min_features=1,
          max_features=5,
          scoring='f1',
          cv=2)

efs.fit(x_train, y_train)



##selected features
efs.best_feature_names_



# selected features ('4', '6', '23', '34'): race_hispanic, race_white, dose_1, loinc_9279-1
x_train.info()



# Fit the estimator using the new feature subset
X_test_efs = efs.transform(x_test)
lr.fit(X_train_efs, y_train)


#predictions on test set
y_pred_test = lr.predict(X_test_efs)

##model evaluation on testing data 
print("Accuracy score for testing data in logit model (EFS) is", accuracy_score(y_test,y_pred_test))
print("Precision score for testing data in logit model (EFS) is", precision_score(y_test,y_pred_test))
print("Recall score for testing data in logit model (EFS) is", recall_score(y_test,y_pred_test))
print("F1 score for testing data in logit model (EFS) is", f1_score(y_test,y_pred_test))


#Confusion Matrix for testing data  
print("Confusion Matrix for testing data in logit model (backward feature selection) is")
pd.DataFrame(confusion_matrix(y_test,y_pred_test),index = ['Actual : 0', 'Actual : 1'],
             columns =['pred : 0', 'pred : 1'])


# # KNN model without pipeline

#scale data
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled=scaler.transform(x_test)



#implement KNN model
knn = KNeighborsClassifier()

#fitting the model
knn.fit(x_train_scaled,y_train)


#getting the predicted values for training and testing data
y_pred_test=knn.predict(x_test_scaled)
y_pred_train=knn.predict(x_train_scaled)


##model evaluation on testing data 
print("Accuracy score for testing data in KNN (before tuning and without pipeline) is", accuracy_score(y_test,y_pred_test))
print("Precision score for testing data in KNN (before tuning and without pipeline) is", precision_score(y_test,y_pred_test))
print("Recall score for testing data in KNN (before tuning and without pipeline) is", recall_score(y_test,y_pred_test))
print("F1 score for testing data in KNN (before tuning and without pipeline) is", f1_score(y_test,y_pred_test))


##Confusion Matrix for testing data 
print("Confusion Matrix for testing data in KNN (before tuning and without pipeline) is")
pd.DataFrame(confusion_matrix(y_test,y_pred_test),index = ['Actual : 0', 'Actual : 1'],
             columns =['pred : 0', 'pred : 1'])



##model evaluation on training data 
print("Accuracy score for training data in KNN (before tuning and without pipeline) is", accuracy_score(y_train,y_pred_train))
print("Precision score for training data in KNN (before tuning and without pipeline) is", precision_score(y_train,y_pred_train))
print("Recall score for training data in KNN (before tuning and without pipeline) is", recall_score(y_train,y_pred_train))
print("F1 score for training data in KNN (before tuning and without pipeline) is", f1_score(y_train,y_pred_train))


##Confusion Matrix for training data  
print("Confusion Matrix for training data in KNN (before tuning and without pipeline) is")
pd.DataFrame(confusion_matrix(y_train,y_pred_train),index = ['Actual : 0', 'Actual : 1'],
             columns =['pred : 0', 'pred : 1'])


# # KNN model with Pipline

#setting up Pipeline

pp=pd.DataFrame(df) 
pipe = Pipeline([('a', MinMaxScaler()), ('b', KNeighborsClassifier())])


# fitting the model on training scaled data
pipe.fit(x_train_scaled, y_train)


##getting predicted values for testing and training data
y_pred_pipe=pipe.predict(x_test_scaled)
y_pred_pipe_train=pipe.predict(x_train_scaled)



##model evaluation on testing data 
print("Accuracy score for testing data in KNN (before tuning and with pipeline) is", accuracy_score(y_test,y_pred_pipe))
print("Precision score for testing data in KNN (before tuning and with pipeline is", precision_score(y_test,y_pred_pipe))
print("Recall score for testing data in KNN (before tuning and with pipeline is", recall_score(y_test,y_pred_pipe))
print("F1 score for testing data in KNN (before tuning and with pipeline) is", f1_score(y_test,y_pred_pipe))

##Confusion Matrix for testing data 
print("Confusion Matrix for testing data in KNN (before tuning and with pipeline) is")
pd.DataFrame(confusion_matrix(y_test,y_pred_pipe),index = ['Actual : 0', 'Actual : 1'],
             columns =['pred : 0', 'pred : 1'])

##model evaluation on training data 
print("Accuracy score for training data in KNN (before tuning and with pipeline) is", accuracy_score(y_train,y_pred_pipe_train))
print("Precision score for training data in KNN (before tuning and with pipeline) is", precision_score(y_train,y_pred_pipe_train))
print("Recall score for training data in KNN (before tuning and with pipeline) is", recall_score(y_train,y_pred_pipe_train))
print("F1 score for training data in KNN (before tuning and with pipeline) is", f1_score(y_train,y_pred_pipe_train))



##Confusion Matrix for training data  
print("Confusion Matrix for training data in KNN (before tuning and with pipeline) is")
pd.DataFrame(confusion_matrix(y_train,y_pred_pipe_train),index = ['Actual : 0', 'Actual : 1'],
             columns =['pred : 0', 'pred : 1'])


# # KNN model Tuned with Pipeline (all 35 X features)

param_grid = { 'b__n_neighbors': range(1,20), 'b__p': [1,10]} 


cv = KFold(n_splits=10, random_state=1, shuffle=True)
grid = GridSearchCV(pipe,param_grid,verbose=3,scoring="f1",cv=10)


grid.fit(x_train,y_train)


grid.cv_results_
grid.best_params_

#best parameters: {'b__n_neighbors': 1, 'b__p': 1}


pipe = Pipeline([('a', MinMaxScaler()), ('b', KNeighborsClassifier(n_neighbors=1,p=1))])


pipe.fit(x_train_scaled, y_train)

##getting predicted values for testing and training data
y_pred_pipe=pipe.predict(x_test_scaled)
y_pred_pipe_train=pipe.predict(x_train_scaled)


##model evaluation on testing data 
print("Accuracy score for testing data in KNN (after tuning and with pipeline) is", accuracy_score(y_test,y_pred_pipe))
print("Precision score for testing data in KNN (after tuning and with pipeline) is", precision_score(y_test,y_pred_pipe))
print("Recall score for testing data in KNN (after tuning and with pipeline) is", recall_score(y_test,y_pred_pipe))
print("F1 score for testing data in KNN (after tuning and with pipeline) is", f1_score(y_test,y_pred_pipe))



##Confusion Matrix for testing data 
print("Confusion Matrix for testing data in KNN (after tuning and with pipeline) is")
pd.DataFrame(confusion_matrix(y_test,y_pred_pipe),index = ['Actual : 0', 'Actual : 1'],
             columns =['pred : 0', 'pred : 1'])



##model evaluation on training data 
print("Accuracy score for training data in KNN (after tuning and with pipeline) is", accuracy_score(y_train,y_pred_pipe_train))
print("Precision score for training data in KNN (after tuning and with pipeline) is", precision_score(y_train,y_pred_pipe_train))
print("Recall score for training data in KNN (after tuning and with pipeline) is", recall_score(y_train,y_pred_pipe_train))
print("F1 score for training data in KNN (after tuning and with pipeline) is", f1_score(y_train,y_pred_pipe_train))


##Confusion Matrix for training data  
print("Confusion Matrix for training data in KNN (after tuning and with pipeline) is")
pd.DataFrame(confusion_matrix(y_train,y_pred_pipe_train),index = ['Actual : 0', 'Actual : 1'],
             columns =['pred : 0', 'pred : 1'])


# # KNN, DT, RFC with 21 backward features with tuned parameters 



#redefining X with only 21 features selected from backward method
x = df[['age','gender_female','race_asian','race_black', 'race_hispanic', 'race_north american native', 
        'race_white', 'race_other','prescription_procurement_duration','active_status_1', 
        'claim_type_DME', 'claim_type_I', 'claim_type_P', 'paid_amount', 'claim_duration', 
        'coverage_duration', 'medicare_payer_type', 'dose_1', 'dose_PRN', 'dose_SCH','loinc_8310-5']]

y= df[['deceased_flag']]

#train test split 70-30
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=100)



'''KNN (pipeline)'''

#initialize and training the model with tuned parameters
pipe = Pipeline([('a', MinMaxScaler()), ('b', KNeighborsClassifier(n_neighbors=1,p=1))])
pipe.fit(x_train, y_train)



##getting predicted values for testing and training data
y_pred_pipe=pipe.predict(x_test)
y_pred_pipe_train=pipe.predict(x_train)


##model evaluation on testing data 
print("Accuracy score for testing data in KNN (tuned + 21 backward features with pipeline) is", accuracy_score(y_test,y_pred_pipe))
print("Precision score for testing data in KNN (tuned + 21 backward features with pipeline) is", precision_score(y_test,y_pred_pipe))
print("Recall score for testing data in KNN (tuned + 21 backward features with pipeline) is", recall_score(y_test,y_pred_pipe))
print("F1 score for testing data in KNN (tuned + 21 backward features with pipeline) is", f1_score(y_test,y_pred_pipe))




##Confusion Matrix for testing data 
print("Confusion Matrix for testing data in KNN (tuned + 21 backward features with pipeline) is")
pd.DataFrame(confusion_matrix(y_test,y_pred_pipe),index = ['Actual : 0', 'Actual : 1'],
             columns =['pred : 0', 'pred : 1'])


##model evaluation on training data 
print("Accuracy score for training data in KNN (tuned + 21 backward features with pipeline) is", accuracy_score(y_train,y_pred_pipe_train))
print("Precision score for training data in KNN (tuned + 21 backward features with pipeline) is", precision_score(y_train,y_pred_pipe_train))
print("Recall score for training data in KNN (tuned + 21 backward features with pipeline) is", recall_score(y_train,y_pred_pipe_train))
print("F1 score for training data in KNN (tuned + 21 backward features with pipeline) is", f1_score(y_train,y_pred_pipe_train))



##Confusion Matrix for training data  
print("Confusion Matrix for training data in KNN (tuned + 21 backward features with pipeline) is")
pd.DataFrame(confusion_matrix(y_train,y_pred_pipe_train),index = ['Actual : 0', 'Actual : 1'],
             columns =['pred : 0', 'pred : 1'])


'''Decision Tree'''

#initialize and training the model with tuned parameters
dt=DecisionTreeClassifier(max_depth=2,min_samples_split=2,random_state=1) 
dt.fit(x_train,y_train) 


##getting predicted values for testing and training data
y_pred_test=dt.predict(x_test)
y_pred_train=dt.predict(x_train)


##model evaluation on testing data 
print("Accuracy score for testing data in DT (tuned + 21 backward features)is", accuracy_score(y_test,y_pred_test))
print("Precision score for testing data in DT (tuned + 21 backward features) is", precision_score(y_test,y_pred_test))
print("Recall score for testing data in DT (tuned + 21 backward features) is", recall_score(y_test,y_pred_test))
print("F1 score for testing data in DT (tuned + 21 backward features) is", f1_score(y_test,y_pred_test))


##Confusion Matrix for training data  
print("Confusion Matrix for training data in DT (tuned + 21 backward features) is")
pd.DataFrame(confusion_matrix(y_test,y_pred_test),index = ['Actual : 0', 'Actual : 1'],
             columns =['pred : 0', 'pred : 1'])



##model evaluation on training data 
print("Accuracy score for training data in DT (tuned + 21 backward features) is", accuracy_score(y_train,y_pred_train))
print("Precision score for training data in DT (tuned + 21 backward features) is", precision_score(y_train,y_pred_train))
print("Recall score for training data in DT (tuned + 21 backward features) is", recall_score(y_train,y_pred_train))
print("F1 score for training data in DT (tuned + 21 backward features) is", f1_score(y_train,y_pred_train))


##Confusion Matrix for training data  
print("Confusion Matrix for training data in DT (tuned + 21 backward features) is")
pd.DataFrame(confusion_matrix(y_train,y_pred_train),index = ['Actual : 0', 'Actual : 1'],
             columns =['pred : 0', 'pred : 1'])


#plotting the tree with tuned parameters
class_names = ['Survived', 'Deceased']
plt.figure(figsize=(15, 15))
plot_tree(dt, class_names=class_names)


x.info()
# x[6] race_white as first split
# used x[14] claim_duration and x[0] age again as with tuning DT on all 35 features



'''RFC'''

rfc=RandomForestClassifier(random_state=1,n_estimators=155)
rfc.fit(x_train,y_train)


##getting predicted values for testing and training data
y_pred_test=rfc.predict(x_test)
y_pred_train=rfc.predict(x_train)



##model evaluation on testing data 
print("Accuracy score for testing data in RFC (tuned + 21 backward features) is", accuracy_score(y_test,y_pred_test))
print("Precision score for testing data in RFC (tuned + 21 backward features) is", precision_score(y_test,y_pred_test))
print("Recall score for testing data in RFC (tuned + 21 backward features) is", recall_score(y_test,y_pred_test))
print("F1 score for testing data in RFC (tuned + 21 backward features) is", f1_score(y_test,y_pred_test))



##Confusion Matrix for testing data 
print("Confusion Matrix for testing data in RFC (tuned + 21 backward features) is")
pd.DataFrame(confusion_matrix(y_test,y_pred_test),index = ['Actual : 0', 'Actual : 1'],
             columns =['pred : 0', 'pred : 1'])


##model evaluation on training data - overfitting
print("Accuracy score for training data in RFC (tuned + 21 backward features) is", accuracy_score(y_train,y_pred_train))
print("Precision score for training data in RFC (tuned + 21 backward features) is", precision_score(y_train,y_pred_train))
print("Recall score for training data in RFC (tuned + 21 backward features) is", recall_score(y_train,y_pred_train))
print("F1 score for training data in RFC (tuned + 21 backward features) is", f1_score(y_train,y_pred_train))


##Confusion Matrix for training data  
print("Confusion Matrix for training data in RFC (tuned + 21 backward features) is")
pd.DataFrame(confusion_matrix(y_train,y_pred_train),index = ['Actual : 0', 'Actual : 1'],
             columns =['pred : 0', 'pred : 1'])


# # KNN, DT, RFC with 8 forward features selction from logit model

#redefining X with only 8 features selected from forward method
x = df[['age','gender_female','race_asian','race_black', 'race_hispanic', 'race_north american native', 
        'race_white', 'coverage_duration']]

y= df[['deceased_flag']]

#train test split 70-30
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=100)



'''KNN (tuned + pipeline)'''


pipe = Pipeline([('a', MinMaxScaler()), ('b', KNeighborsClassifier(n_neighbors=1,p=1))])
pipe.fit(x_train, y_train)



##getting predicted values for testing and training data
y_pred_pipe=pipe.predict(x_test)
y_pred_pipe_train=pipe.predict(x_train)


##model evaluation on testing data 
print("Accuracy score for testing data in KNN (tuned + 8 forward features with pipeline) is", accuracy_score(y_test,y_pred_pipe)) 
print("Precision score for testing data in KNN (tuned + 8 forward features with pipeline) is", precision_score(y_test,y_pred_pipe))
print("Recall score for testing data in KNN (tuned + 8 forward features with pipeline) is", recall_score(y_test,y_pred_pipe))
print("F1 score for testing data in KNN (tuned + 8 forward features with pipeline) is", f1_score(y_test,y_pred_pipe))


##Confusion Matrix for testing data 
print("Confusion Matrix for testing data in KNN (tuned + 8 forward features with pipeline) is")
pd.DataFrame(confusion_matrix(y_test,y_pred_pipe),index = ['Actual : 0', 'Actual : 1'],
             columns =['pred : 0', 'pred : 1'])


##model evaluation on training data 
print("Accuracy score for training data in KNN (tuned + 8 forward features with pipeline) is", accuracy_score(y_train,y_pred_pipe_train))
print("Precision score for training data in KNN (tuned + 8 forward features with pipeline) is", precision_score(y_train,y_pred_pipe_train))
print("Recall score for training data in KNN (tuned + 8 forward features with pipeline) is", recall_score(y_train,y_pred_pipe_train))
print("F1 score for training data in KNN (tuned + 8 forward features with pipeline) is", f1_score(y_train,y_pred_pipe_train))



##Confusion Matrix for training data  
print("Confusion Matrix for training data in KNN (tuned + 8 forward features with pipeline) is")
pd.DataFrame(confusion_matrix(y_train,y_pred_pipe_train),index = ['Actual : 0', 'Actual : 1'],
             columns =['pred : 0', 'pred : 1'])



'''Decision Tree'''

#initialize and training the model with tuned parameters
dt=DecisionTreeClassifier(max_depth=2,min_samples_split=2,random_state=1) 
dt.fit(x_train,y_train) 



##getting predicted values for testing and training data
y_pred_test=dt.predict(x_test)
y_pred_train=dt.predict(x_train)


##model evaluation on testing data 
print("Accuracy score for testing data in DT (tuned + 8 forward features)is", accuracy_score(y_test,y_pred_test)) #0.8684
print("Precision score for testing data in DT (tuned + 8 forward features) is", precision_score(y_test,y_pred_test)) #0.2
print("Recall score for testing data in DT (tuned + 8 forward features) is", recall_score(y_test,y_pred_test)) #0.5
print("F1 score for testing data in DT (tuned + 8 forward features) is", f1_score(y_test,y_pred_test)) #0.2857


##Confusion Matrix for testing data  
print("Confusion Matrix for testing data in DT (tuned + 8 forward features) is")
pd.DataFrame(confusion_matrix(y_test,y_pred_test),index = ['Actual : 0', 'Actual : 1'],
             columns =['pred : 0', 'pred : 1'])



##model evaluation on training data 
print("Accuracy score for training data in DT (tuned + 8 forward features) is", accuracy_score(y_train,y_pred_train)) #0.93
print("Precision score for training data in DT (tuned + 8 forward features) is", precision_score(y_train,y_pred_train)) #0.7
print("Recall score for training data in DT (tuned + 8 forward features) is", recall_score(y_train,y_pred_train)) #0.7
print("F1 score for training data in DT (tuned + 8 forward features) is", f1_score(y_train,y_pred_train)) #0.7



##Confusion Matrix for training data  
print("Confusion Matrix for training data in DT (tuned + 8 forward features) is")
pd.DataFrame(confusion_matrix(y_train,y_pred_train),index = ['Actual : 0', 'Actual : 1'],
             columns =['pred : 0', 'pred : 1'])



#plotting the tree with tuned parameters
class_names = ['Survived', 'Deceased']
plt.figure(figsize=(15, 15))
plot_tree(dt, class_names=class_names)



x.info()
# x[6] race_white as first split followed by x[0]age only


'''RFC'''

#initialize and training the model with tuned parameters
rfc=RandomForestClassifier(random_state=1,n_estimators=155)
rfc.fit(x_train,y_train)



##getting predicted values for testing and training data
y_pred_test=rfc.predict(x_test)
y_pred_train=rfc.predict(x_train)




##model evaluation on testing data 
print("Accuracy score for testing data in RFC (tuned + 8 forward features) is", accuracy_score(y_test,y_pred_test))  #0.86
print("Precision score for testing data in RFC (tuned + 8 forward features) is", precision_score(y_test,y_pred_test)) #0.2
print("Recall score for testing data in RFC (tuned + 8 forward features) is", recall_score(y_test,y_pred_test)) #0.5
print("F1 score for testing data in RFC (tuned + 8 forward features) is", f1_score(y_test,y_pred_test)) #0.28

##Confusion Matrix for testing data 
print("Confusion Matrix for testing data in RFC (tuned + 8 forward features) is")
pd.DataFrame(confusion_matrix(y_test,y_pred_test),index = ['Actual : 0', 'Actual : 1'],
             columns =['pred : 0', 'pred : 1'])



##model evaluation on training data (overfitting)
print("Accuracy score for training data in RFC (tuned + 8 forward features) is", accuracy_score(y_train,y_pred_train)) #1.0
print("Precision score for training data in RFC (tuned + 8 forward features) is", precision_score(y_train,y_pred_train)) #1.0
print("Recall score for training data in RFC (tuned + 8 forward features) is", recall_score(y_train,y_pred_train)) #1.0
print("F1 score for training data in RFC (tuned + 8 forward features) is", f1_score(y_train,y_pred_train)) #1.0

##Confusion Matrix for training data  
print("Confusion Matrix for training data in RFC (tuned + 8 forward features) is")
pd.DataFrame(confusion_matrix(y_train,y_pred_train),index = ['Actual : 0', 'Actual : 1'],
             columns =['pred : 0', 'pred : 1'])

