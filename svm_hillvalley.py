# -*- coding: utf-8 -*-
"""
Created on Fri Dec 04 19:12:14 2015

@author: sai
"""
import numpy as np
import array
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn import cross_validation
from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import train_test_split
data=np.genfromtxt("Hill_Valley_Training.data.txt", delimiter=",", comments="#")
X,y = data[1:,:-1], data[1:,-1]
print X.shape #100 features
train_X,test_X,train_y,test_y= train_test_split(X, y, test_size=0.20, random_state=42)
Accuracy=[] #0.475
classifier = OneVsRestClassifier(SVC(C=1, kernel='rbf', gamma=0.5)).fit(train_X, train_y)
y_predict = classifier.predict(test_X)
Accuracy = metrics.accuracy_score(test_y, y_predict)
print Accuracy
