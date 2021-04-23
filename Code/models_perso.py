#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 09:22:57 2021

@author: Celian
based on Remi Viola Work
https://github.com/RemiViola/gamma-kNN
"""


from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator, ClassifierMixin
from metric_learn import LMNN
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier

import numpy as np

OS_methods={
    "SMOTE":SMOTE(),
    "B_SMOTE":BorderlineSMOTE(),
    "ADASYN":ADASYN(),
    "ENN_SMOTE":EditedNearestNeighbours(),
    "TL_Smoke":SMOTETomek()
}
class SimpleKnn(BaseEstimator, ClassifierMixin):
    def __init__(self, nb_nn = 3):
        self.nb_nn = nb_nn
        print("INIT Knn simple")
    
    def fit(self, X, y):
        self.dim_ = len(X[0])
        self.X_ = X
        self.y_ = y
        # if OS_str:     
        #     try:
        #         OS=OS_methods[OS_str]       
        #     except ValueError:
        #         print("not a valid over_sampling method")
                
        #     self.OS_ = OS
        #     X_os,y_os = self.OS_.fit_resample(self.X_,self.y_)
            
            
        #     self.model = KNeighborsClassifier(n_neighbors = self.nb_nn)
        #     self.model.fit(X_os,y_os)
        # else:

        self.model = KNeighborsClassifier(n_neighbors = self.nb_nn)
        self.model.fit(self.X_,self.y_)
        
        
        
    def predict(self, X):          
        return  self.model.predict(X)
    
class wKnn(BaseEstimator, ClassifierMixin):
    def __init__(self, nb_nn = 3):
        self.nb_nn = nb_nn
        print("INIT Knn simple")
    
    def fit(self, X, y):
        self.dim_ = len(X[0])
        self.X_ = X
        self.y_ = y
        
        # if OS_str:     
        #     try:
        #         OS=OS_methods[OS_str]       
        #     except ValueError:
        #         print("not a valid over_sampling method")
                
        #     self.OS_ = OS
        #     X_os,y_os = self.OS_.fit_resample(self.X_,self.y_)
            
            
        #     self.model = KNeighborsClassifier(n_neighbors = self.nb_nn, weights="distance")
        #     self.model.fit(X_os,y_os)
        # else:

        self.model = KNeighborsClassifier(n_neighbors = self.nb_nn,metric='euclidean', weights="distance")
        self.model.fit(self.X_,self.y_)
        
        return self

    def predict(self, X):          
        return  self.model.predict(X)

class LMNN_perso(BaseEstimator, ClassifierMixin):
    def __init__(self, nb_nn = 3,lr=0.1):
        self.nb_nn = nb_nn
        self.lr=lr
    
    def fit(self, X, y):
        self.dim_ = len(X[0])
        self.X_ = X
        self.y_ = y
        
        lmnn = LMNN(k=self.nb_nn,learn_rate=self.lr)
        lmnn.fit(self.X_, self.y_)
        tx = lmnn.transform(self.X_)
        # print(tx)
        self.model = KNeighborsClassifier(n_neighbors = self.nb_nn)
        self.model.fit(tx,self.y_)
        return self

    def predict(self, X):          
        return  self.model.predict(X)

class dupKnn(BaseEstimator, ClassifierMixin):
    def __init__(self, nb_nn = 3):
        self.nb_nn = nb_nn
    
    def fit(self, X, y):
        self.dim_ = len(X[0])
        self.X_ = X
        self.y_ = y
        
        ros = RandomOverSampler(random_state=0)
        X_resampled, y_resampled = ros.fit_resample(X, y)

        self.model = KNeighborsClassifier(n_neighbors = self.nb_nn)
        self.model.fit(X_resampled,y_resampled)
        return self

    def predict(self, X):          
        return  self.model.predict(X)

class GammaKnn(BaseEstimator, ClassifierMixin):
    def __init__(self, nb_nn = 3, gamma = 0.5):
        self.gamma = gamma
        self.nb_nn = nb_nn

    
    def fit(self, X, y):
        self.dim_ = len(X[0])
        self.X_ = X
        self.y_ = y

       
        self.nn_pos_ = NearestNeighbors(n_neighbors = self.nb_nn)
        self.nn_pos_.fit(self.X_[self.y_ == 1])
    
        self.nn_neg_ = NearestNeighbors(n_neighbors = self.nb_nn)
        self.nn_neg_.fit(self.X_[self.y_ != 1])

        return self
        
    def predict(self, X): 
        distance_test_to_positive = self.nn_pos_.kneighbors(X, return_distance = True)[0]
        distance_test_to_negative = self.nn_neg_.kneighbors(X, return_distance = True)[0]
        

        return  [np.count_nonzero((np.argsort(np.concatenate((distance_test_to_positive[i]*self.gamma,distance_test_to_negative[i])))<self.nb_nn)[:self.nb_nn])>=(self.nb_nn//2+1) for i in range(len(X))]

class GammaKnnOS(BaseEstimator, ClassifierMixin):
    def __init__(self, gamma_real = 0.5, gamma_synth = 0.5, nb_nn = 3,OS=SMOTE()):
        self.gamma_real = gamma_real
        self.gamma_synth = gamma_synth
        self.nb_nn = nb_nn
        self.OS=OS
        # print(OS)
        # try:
        #     self.OS=OS_methods[OS]       
        # except ValueError:
        #     print("not a valid over_sampling method")
    
    def fit(self, X, y ):
        self.dim_ = len(X[0])
        self.X_ = X
        self.y_ = y
        
        
    
        X_os,y_os = self.OS.fit_resample(self.X_,self.y_)
        
        # Real/Synth Split
        
        index_synth = list(range(len(y_os)))
        for i in range(len(self.y_)):
            index_synth = [i_ for i_ in index_synth if np.any(X_os[i_] != self.X_[i])]
            
        index_real = list(set(list(range(len(y_os))))-set(index_synth))
        
        X_synth, y_synth = X_os[index_synth], y_os[index_synth]
        X_real, y_real = X_os[index_real], y_os[index_real]
        
        # Gamma k-NN
        
        self.nn_pos_real_ = NearestNeighbors(n_neighbors = self.nb_nn)
        self.nn_pos_real_.fit(X_real[y_real == 1])
                        
        self.nn_pos_synth_ = NearestNeighbors(n_neighbors = self.nb_nn)
        self.nn_pos_synth_.fit(X_synth[y_synth == 1])
        
        self.nn_neg_ = NearestNeighbors(n_neighbors = self.nb_nn)
        self.nn_neg_.fit(X_os[y_os != 1])
        
        return self
    
    def predict(self, X):        
        distance_test_to_real_positive = self.nn_pos_real_.kneighbors(X, return_distance = True)[0]
        distance_test_to_synth_positive = self.nn_pos_synth_.kneighbors(X, return_distance = True)[0]
        distance_test_to_negative = self.nn_neg_.kneighbors(X, return_distance = True)[0]

        return [np.count_nonzero((np.argsort(np.concatenate((distance_test_to_real_positive[i]*self.gamma_real,distance_test_to_synth_positive[i]*self.gamma_synth,distance_test_to_negative[i])))<self.nb_nn*2)[:self.nb_nn])>=(self.nb_nn//2+1) for i in range(len(X))]
        