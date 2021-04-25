#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 21:46:55 2021

@author: dobby
"""

import numpy as np
import pylab as pl
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
from sklearn import svm
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

X_train = shuffle(pd.read_csv("/content/drive/feature_Train1.csv",header=None))
train2 = shuffle(pd.read_csv("/content/drive/feature_Train2.csv",header=None))
train3 = shuffle(pd.read_csv("/content/drive/feature_Train3.csv",header=None))
train4 = shuffle(pd.read_csv("/content/drive/feature_Train4.csv",header=None))
#X_train = X_train.append(train2)
#X_train = X_train.append(train3)
#X_train = X_train.append(train4)
Y_train = shuffle(pd.read_csv("/content/drive/att_Train.csv",header=None))
X_test = shuffle(pd.read_csv("/content/drive/feature_Test.csv",header=None))
Y_test = shuffle(pd.read_csv("/content/drive/att_Test.csv",header=None))
print("Any missing sample in training set:",X_train.isnull().values.any())
print("Any missing sample in test set:",X_test.isnull().values.any(), "\n")

train = X_train
total = train.append(train2)
total1 = total.append(train3)
total2 = total1.append(train4)
X_train = total2
Y_train_ori = Y_train
Y_test_ori = Y_test

for i in range(3,6):

  Y_train_label = Y_train_ori.iloc[:,i]
  Y_test_label = Y_test_ori.iloc[:,i]

# Dimension of Train and Test set 
  print("Dimension of Train set",X_train.shape)
  print("Dimension of Test set",X_test.shape,"\n")
  print("i = ",i)
  # Transforming non numerical labels into numerical labels

  encoder = preprocessing.LabelEncoder()

  # encoding train labels 
  encoder.fit(Y_train_label)
  Y_train = encoder.transform(Y_train_label)

  # encoding test labels 
  encoder.fit(Y_test_label)
  Y_test = encoder.transform(Y_test_label)

  #Total Number of Continous and Categorical features in the training set
  num_cols = X_train._get_numeric_data().columns
  print("Number of numeric features:",num_cols.size)
  #list(set(X_train.columns) - set(num_cols))


  names_of_predictors = list(X_train.columns.values)

  # Scaling the Train and Test feature set 
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)

  svm_model = svm.SVC()
  svm_model.fit(X_train_scaled, Y_train)
# View the accuracy score
#print('Best score for training data:', svm_model.best_score_,"\n") 
#C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    #decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
    #max_iter=-1, probability=False, random_state=None, shrinking=True,
    #tol=0.001, verbose=False
# View the best parameters for the model found using grid search
#print('Best C:',svm_model.best_estimator_.C,"\n") 
#print('Best Kernel:',svm_model.best_estimator_.kernel,"\n")
#print('Best Gamma:',svm_model.best_estimator_.gamma,"\n")

#final_model = svm_model.best_estimator_
  Y_pred = svm_model.predict(X_test_scaled)
  Y_pred_label = list(encoder.inverse_transform(Y_pred))
  Y_pred_att1 = Y_pred_label
# Making the Confusion Matrix
#print(pd.crosstab(Y_test_label, Y_pred_label, rownames=['Actual Activity'], colnames=['Predicted Activity']))
  print(confusion_matrix(Y_test_label,Y_pred_label))
  print("\n")
  print(classification_report(Y_test_label,Y_pred_label))

  print("Training set score for SVM: %f" % svm_model.score(X_train_scaled , Y_train))
  print("Testing  set score for SVM: %f" % svm_model.score(X_test_scaled  , Y_test ))
