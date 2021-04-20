# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 16:22:57 2021

@author: Celian
"""

import pandas as pd
from sklearn import preprocessing

############ FIRST TRY OF IMPLEMENTATION DATASET CLEANING PART

class dataset():
    def __init__(self, file_path, to_predict_col_idx=-1, focused_pred_val=None):
        self.file_path=file_path
        self.data = self.load_dataset()
        self.size = self.data.shape[0]
        self.dim = self.data.shape[1]
        self.to_predict_col=self.get_pred_col(to_predict_col_idx,focused_pred_val)
        # print(focused_pred_val)
        stat=dict(self.to_predict_col.value_counts())
        # print(stat)
        self.pos_part=stat[True]/self.size*100
        self.neg_part=100-self.pos_part
        self.normalized=self.get_normalized(to_predict_col_idx)


    def load_dataset(self):
        return pd.read_csv(self.file_path, header=None)

    def get_pred_col(self,to_predict_col,focused_pred_val):
        df=self.data
        if focused_pred_val==None:
            return df[df.columns[to_predict_col]]
        else:
            return df[df.columns[to_predict_col]].astype(str) == str(focused_pred_val)

    def describe(self):
        r_n_part=round(self.neg_part,2)
        r_p_part=round(self.pos_part,2)
        r_IR=round(self.neg_part/r_p_part,2)

        print({"size":self.size,"dim":self.dim,"pos_part":r_p_part,"neg_part":r_n_part,"IR": r_IR})  

    def get_normalized(self,to_predict_col_idx):
    	#### TO DO : how to deal with categorical data
        temp=self.data.iloc[: , :to_predict_col_idx]
        x = temp.values #returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df = pd.DataFrame(x_scaled)
        return df