#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

dummycl = DummyClassifier(strategy="most_frequent")
gmb = GaussianNB()
dectree = tree.DecisionTreeClassifier()
rdforest = RandomForestClassifier()
logreg = LogisticRegression(solver="liblinear")
svc = svm.SVC(gamma='scale')

lst_classif = [dummycl, gmb, dectree, rdforest, logreg, svc]
lst_classif_names = ['Dummy', 'Naive Bayes', 'Decision tree', 'Random Forest', 'Logistic regression', 'SVM']

def accuracy_score(lst_classif,lst_classif_names,X,y):
    for clf,name_clf in zip(lst_classif,lst_classif_names):
        scores = cross_val_score(clf, X, y, cv=5)
        print("Accuracy of "+name_clf+" classifier on cross-validation: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

def confusion_matrix(lst_classif,lst_classif_names,X,y):
    for clf,name_clf in zip(lst_classif,lst_classif_names):
        predicted = cross_val_predict(clf, X, y, cv=5) 
        print("Accuracy of "+name_clf+" classifier on cross-validation: %0.2f" % metrics.accuracy_score(y, predicted))
        print(metrics.confusion_matrix(y, predicted))




#data = pd.read_csv('bank-marketing/bank-additional-full.csv')
data = pd.read_csv('bank-marketing/bank.test.csv')
#data = pd.read_csv('bank-marketing/bank.train.csv')
#print(data.shape)
#print(data.dtypes)

        
# Replace missing values by mean and scale numeric values
data_num = data.select_dtypes(include='float64')
scaler = StandardScaler()
#data_num = scaler.fit_transform(data_num)
#data_num = scaler.transform(data_num)




# Replace missing values by mean and discretize categorical values
#data_cat = data.select_dtypes(exclude='float64').drop('class',axis=1)


# Disjonction with OneHotEncoder
#from sklearn.preprocessing import OneHotEncoder
#encoder = OneHotEncoder(handle_unknown="ignore")
# encoder.fit(X_cat)
# X_cat = encoder.transform(X_cat).toarray()


