#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 09:22:57 2021

@author: Celian
based on Remi Viola Work
https://github.com/RemiViola/gamma-kNN
"""


from sklearn.neighbors import NearestNeighbors

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import EditedNearestNeighbours

import numpy as np

OS_methods={
    "SMOTE":SMOTE(),
    "B_SMOTE":BorderlineSMOTE(),
    "ADASYN":ADASYN(),
    "ENN_SMOTE":EditedNearestNeighbours(),
    "TL_Smoke":SMOTETomek()
}
class SimpleKnn():
    def __init__(self, nb_nn = 3):
        self.nb_nn = nb_nn
        print("INIT")
    
    def fit(self, X, y,OS_str):
        self.dim_ = len(X[0])
        self.X_ = X
        self.y_ = y
        print("FIT")

        print("'os :'",OS_str)
        if(OS_str): 
            try:
                OS=OS_methods[OS_str[0]]       
            except ValueError:
                print("not a valide over_sampling method")
            
            self.OS_ = OS
            X_os,y_os = self.OS_.fit_resample(self.X_,self.y_)
            print(X_os)
            self.nn_pos_ = NearestNeighbors(n_neighbors = self.nb_nn)
            print("hey")
            self.nn_pos_.fit(X_os[y_os == 1])
            
            self.nn_neg_ = NearestNeighbors(n_neighbors = self.nb_nn)
            self.nn_neg_.fit(X_os[y_os != 1])
        else:
            self.nn_pos_ = NearestNeighbors(n_neighbors = self.nb_nn)
            self.nn_pos_.fit(self.X_[self.y_ == 1])
        
            self.nn_neg_ = NearestNeighbors(n_neighbors = self.nb_nn)
            self.nn_neg_.fit(self.X_[self.y_ != 1])

        print("finish")
        return self
        
    def predict(self, X):            

        print("predict")
        distance_test_to_positive = self.nn_pos_.kneighbors(X, return_distance = True)[0]
        distance_test_to_negative = self.nn_neg_.kneighbors(X, return_distance = True)[0]
        

        return  knn.predict(X)
    
        
       
class Gamma():
    def __init__(self, nb_nn = 3, gamma = 0.5):
        self.gamma = gamma
        self.nb_nn = nb_nn
        print("INIT")
    
    def fit(self, X, y,OS_str):
        self.dim_ = len(X[0])
        self.X_ = X
        self.y_ = y
        print("FIT")

        print("'os :'",OS_str)
        if(OS_str): 
            try:
                OS=OS_methods[OS_str[0]]       
            except ValueError:
                print("not a valide over_sampling method")
            
            self.OS_ = OS
            X_os,y_os = self.OS_.fit_resample(self.X_,self.y_)
            print(X_os)
            self.nn_pos_ = NearestNeighbors(n_neighbors = self.nb_nn)
            print("hey")
            self.nn_pos_.fit(X_os[y_os == 1])
            
            self.nn_neg_ = NearestNeighbors(n_neighbors = self.nb_nn)
            self.nn_neg_.fit(X_os[y_os != 1])
        else:
            self.nn_pos_ = NearestNeighbors(n_neighbors = self.nb_nn)
            self.nn_pos_.fit(self.X_[self.y_ == 1])
        
            self.nn_neg_ = NearestNeighbors(n_neighbors = self.nb_nn)
            self.nn_neg_.fit(self.X_[self.y_ != 1])

        print("finish")
        return self
        
    def predict(self, X):            
        print("predict")
        distance_test_to_positive = self.nn_pos_.kneighbors(X, return_distance = True)[0]
        distance_test_to_negative = self.nn_neg_.kneighbors(X, return_distance = True)[0]
        

        return  [np.count_nonzero((np.argsort(np.concatenate((distance_test_to_positive[i]*self.gamma,distance_test_to_negative[i])))<self.nb_nn)[:self.nb_nn])>=(self.nb_nn//2+1) for i in range(len(X))]

    
        
        