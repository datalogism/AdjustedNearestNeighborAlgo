#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 09:22:57 2021

@author: Celian
based on Remi Viola Work
https://github.com/RemiViola/MLFP/blob/master/Code/MLFP.py
"""

import argparse
import time
import os

from functions import data_recovery

from models_perso import GammaKnn

# Metric Learning tools
# from skopt.space import Real
# from skopt import gp_minimize

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import numpy as np

import pandas as pd

from multiprocessing import Pool

import warnings
from sklearn.exceptions import DataConversionWarning

# UserWarning: The objective has been evaluated at this point before.
warnings.filterwarnings("ignore", category=UserWarning)

# DataConversionWarning: Data with input dtype int64 was converted to float64
# by StandardScaler.
warnings.filterwarnings("ignore", category=DataConversionWarning)

beta = 1  # If we want change the f-measure


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="autompg", help=('dataset \
                    in"satimage","pageblocks","abalone20","abalone17",\
                    "abalone8","segmentation","wine4","yeast6","yeast3",\
                    "german","vehicle","pima","balance","autompg","libras",\
                    "iono","glass","wine","hayes"'))

parser.add_argument('--seed', type=int, default=123, help='seed for the \
                    randomness, in [12,123,1234,12345,123456]')

parser.add_argument('--nb_nn', type=int, default=1, help='number of neighbors \
                    for kNN')

#MLFP specific
# parser.add_argument('--mu', nargs=2, type=float, default=[0.0, 1.0], help='min\
#                     and max values for mu on MLFP')

# parser.add_argument('--c', nargs=2, type=float, default=[0.0, 1.0], help='min \
#                     and max values for c on MLFP')

#gamma specific
# IF WE WANT TO FINETUNE IT
# parser.add_argument('--gamma', nargs=2, type=float, default=[0.0, 1.0], help='min \
#                     and max values for gammaK-nn')
parser.add_argument('--gamma', nargs=1, type=float, default=1, help='min \
                    and max values for gammaK-nn')

parser.add_argument('--os', nargs=1, type=str, default=None,help=('oversampling strategies are \
                    in "SMOTE","B_SMOTE","ADASYN","ENN_SMOTE","TL_Smoke"'))

parser.add_argument('--nb_cv', type=int, default=0, help='number of fold for \
                    the cross validation')

parser.add_argument('--normalization', type=str2bool, default=True, help='\
                    perform a normalization')

parser.add_argument('--pca', type=str2bool, default=True, help='perform a PCA\
                    for dimensonality reduction')

#MLFP specific
# parser.add_argument('--diag', type=str2bool, default=True, help='constrain L \
#                     to be diagonal')

# parser.add_argument('--cons1', type=str2bool, default=True, help='constrain \
#                     eigenvalues of L to be less than one')

opt = parser.parse_args()

np.random.seed(seed=opt.seed)

print(opt)

date = time.strftime("%Y_%b_%d_%H_%M_%S", time.localtime(round(time.time())))

print(date)

##########################
#                        #
# File Names Preparation #
#                        #
##########################

opt.name = str(opt.nb_nn) + 'NN'

if opt.os:
    opt.name = opt.name + str(opt.os)
else:
    opt.name = opt.name + '_nonos'

if opt.normalization:
    opt.name = opt.name + '_norm'
else:
    opt.name = opt.name + '_nonnorm'

if opt.pca:
    opt.name = opt.name + '_PCA'

# if opt.diag:
#     opt.name = opt.name + '_diag'
# else:
#     opt.name = opt.name + '_nondiag'

# if opt.cons1:
#     opt.name = opt.name + '_cons1'
# else:
#     opt.name = opt.name + '_noncons1'

opt.name = opt.name + f'_{opt.dataset}_{opt.seed}'

###################
#                 #
# Folder Creation #
#                 #
###################

os.makedirs(f'../Plots/', exist_ok=True)

os.makedirs(f'../Outputs/', exist_ok=True)

#############
#           #
# Functions #
#           #
#############


def learning(param, ctrl=False):

    params, X_tr, y_tr, X_va, y_va = param
    X_te = X[test_index]
    y_te = y[test_index]

    #################
    #               #
    # Normalization #
    #               #
    #################

    if opt.normalization:
        normalizer = StandardScaler()
        normalizer.fit(X_tr)
        X_tr = normalizer.transform(X_tr)
        X_va = normalizer.transform(X_va)
        X_te = normalizer.transform(X_te)

    #######
    #     #
    # PCA #
    #     #
    #######

    if opt.pca:
        pca = PCA(n_components=dim)
        pca.fit(X_tr)
        nb_pca = np.where(
                (np.cumsum(pca.explained_variance_ratio_) > 0.999) == 1
                )[0][0]+1
        nb_pca = 2
        pca = PCA(n_components=nb_pca)
        pca.fit(X_tr)
        X_tr = pca.transform(X_tr)
        X_va = pca.transform(X_va)
        X_te = pca.transform(X_te)

    gammaKnn = GammaKnn(nb_nn=opt.nb_nn, gamma=params)
    # For controling overfitting
    gammaKnn.ctrl = ctrl

    if gammaKnn.ctrl:
        gammaKnn.X_train = X_tr
        gammaKnn.X_valid = X_va
        gammaKnn.y_train = y_tr
        gammaKnn.y_valid = y_va
        gammaKnn.X_test = X_te
        gammaKnn.y_test = y_te
        gammaKnn.store_f_m = []
        gammaKnn.store_loss = []

    gammaKnn.fit(X_tr, y_tr,opt.os)

    # if gammaKnn.ctrl:
    #     df = pd.DataFrame(gammaKnn.store_loss)
    #     df.columns = ['Train', 'Valid', 'Test']
    #     df.to_csv(f'../Outputs/{opt.name}_loss.csv')
    #     df = pd.DataFrame(gammaKnn.store_L)
    #     df.to_csv(f'../Outputs/{opt.name}_all_L.csv')
    # if gammaKnn.ctrl:
    #     gammaKnn.ctrl = False

    pred = gammaKnn.predict(X_va)

    TN, FP, FN, TP = confusion_matrix(y_va, pred).ravel()

    f_measure = (1+beta**2)*TP / ((1+beta**2)*TP+(beta**2)*FN+FP)
    
    if ctrl:
        # df = pd.DataFrame(gammaKnn.store_f_m)
        # df.columns = ['Train', 'Valid', 'Test']
        # df.to_csv(f'../Outputs/{opt.name}_ctrl_f_measure.csv')

        df = pd.DataFrame(pred)
        df.to_csv(f'../Outputs/{opt.name}_prediction.csv')

        # if opt.pca:
        #     import matplotlib.pyplot as plt
        #     from matplotlib.colors import ListedColormap

        #     cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        #     cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

        #     h = 0.05
        #     x_min, x_max = X_tr[:, 0].min()-1, X_tr[:, 0].max()+1
        #     y_min, y_max = X_tr[:, 1].min()-1, X_tr[:, 1].max()+1

        #     xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
        #                          np.arange(y_min, y_max, h))
        #     z = gammaKnn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        #     plt.figure()
        #     plt.pcolormesh(xx, yy, z, cmap=cmap_light)
        #     plt.scatter(X_tr[:, 0], X_tr[:, 1], c=y_tr, cmap=cmap_bold, s=.1)
        #     plt.xlabel(r'$x_1$')
        #     plt.ylabel(r'$x_2$')

        #     plt.savefig(f'../Plots/{opt.name}_classifier.pdf')
        #     plt.close()

        return [round(f_measure, 3), TN, FP, FN, TP, gammaKnn]
 
    return f_measure, TN, FP, FN, TP, gammaKnn


def objective(params):

    with Pool() as pool:
        f_measure = pool.map_async(
                learning,
                [(params,
                  X[train_index][trainv_index],
                  y[train_index][trainv_index],
                  X[train_index][valid_index],
                  y[train_index][valid_index],
                  ) for trainv_index, valid_index in kf.split(
                          X[train_index],
                          y[train_index])]).get()

    f_measure = np.mean(f_measure)
    return 1 - f_measure


#################
#               #
# Data Recovery #
#               #
#################

X, y, dim = data_recovery(opt, date)
# ADDED CODE FOR DESCRIBE Dataset
# size = X.shape[0]
# unique, counts = np.unique(y, return_counts=True)
# stat=dict(zip(unique, counts))
# pos_part=stat[True]/size*100
# neg_part=100-pos_part
# r_n_part=round(neg_part,2)
# r_p_part=round(pos_part,2)
# r_IR=round(neg_part/r_p_part,2)
# print({"size":size,"dim":dim,"pos_part":r_p_part,"neg_part":r_n_part,"IR": r_IR})

####################
#                  #
# Train Test Split #
#                  #
####################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=opt.seed)


# mlfp specific ? 
kf = StratifiedKFold(n_splits=5, random_state=opt.seed, shuffle=True)
kf = kf.split(X_train, y_train)
train_index, test_index = kf.__next__()

########
#      #
# MLFP #
#      #
########

######
#    #
# CV #
#    #
######
##  HERE IF WE WANT TO FINETUNE HYPERPARAMETER
# kf = StratifiedKFold(n_splits=opt.nb_cv, random_state=opt.seed, shuffle=True)

# space = [Real(opt.gamma[0], opt.gamma[1], name='mu')]

# res_gp = gp_minimize(objective, space, n_calls=400, random_state=opt.seed,
#                      n_jobs=-1)

# gamma = res_gp.x[0]

############
#          #
# Learning #
#          #
############
gamma = opt.gamma

f_measure, TN, FP, FN, TP, gml = learning((gamma,
                                           X[train_index], y[train_index],
                                           X[test_index], y[test_index]),ctrl=True)


print(f'MLFP -> {f_measure} ( gamma = {gamma} )')
pd.DataFrame([f_measure, TN, FP, FN, TP, gamma]).to_csv(
        f'../Outputs/{opt.name}_f_measure_TN_FP_FN_TP_gamma.csv')

date = time.strftime("%Y_%b_%d_%H_%M_%S", time.localtime(round(time.time())))

print(date)
