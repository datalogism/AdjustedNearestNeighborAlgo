# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 16:04:36 2021

@author: Celian
"""
import os
 
f_dir="C:/Users/Celian/Desktop/THESE3/SaintEtienne/Exo/AdjustedNearestNeighborAlgo/datasets/vehicule"
files = os.listdir(f_dir)
print(arr)

for fname in files:
    if(".dat" in fname):
        

import pandas as pd
f_dir="C:/Users/Celian/Desktop/THESE3/SaintEtienne/Exo/AdjustedNearestNeighborAlgo/datasets/glass/"

dataset = pd.read_csv(f_dir+'glass.data')
dataset.shape
dataset.head()