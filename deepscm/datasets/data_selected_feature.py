# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 13:38:19 2021

@author: Friso Gerben Heslinga
"""
import pandas as pd
import numpy as np
import os

main_path = r'C:\Users\shiqi\PycharmProjects\DSCM_fundus\deepscm\deepscm\datasets\original_split_dataset'
features_dir = os.path.join(main_path, 'T2D_features_combined_20210921.xlsx')
train_dir = os.path.join(main_path, 'train_features.xlsx')
test_dir = os.path.join(main_path, 'test_features.xlsx')
val_dir = os.path.join(main_path, 'val_features.xlsx')
test_newT2D = os.path.join(main_path, 'test2_features.xlsx')

def preprocess_feature(dir, dir2):
    all_features = pd.read_excel(dir, engine='openpyxl', dtype=str)
    selected_features= ['Filename', 'RandomID', 'orig_path_dir', 'Age', 'SEX', 'N_GTS_WHO', 'N_HT', 'bmi', 'waist','smoking_3cat']
    selected_features = all_features[selected_features]
    selected_features= selected_features.rename({'orig_path_dir':'path', 'Age': 'age',  'SEX': 'sex', \
                                                 'N_GTS_WHO': 'T2D', 'N_HT': 'HT', 'smoking_3cat':'smoking'}, axis=1)
    selected_features = selected_features.dropna()
    dtype_dic = {'Filename': 'str', 'RandomID':'str', 'path':'str', 'age': 'int', 'sex': 'int', \
                 'T2D': 'int', 'HT': 'int', 'bmi':'float', 'waist':'float', 'smoking': 'int'}

    selected_features = selected_features.astype(dtype_dic).copy()
    selected_features = selected_features.loc[selected_features['T2D'].isin([0,3])]
    selected_features["sex"] = selected_features["sex"].map({2:0, 1:1})
    selected_features["T2D"] = selected_features["T2D"].map({0:0, 3:1})
    selected_features["path"] = selected_features["path"].str.replace('\\', '/')
    selected_features.to_excel(dir2)

dir_list = [train_dir, val_dir, test_dir, test_newT2D]
for dir in dir_list:
    preprocess_feature(dir, dir.replace('original_split_dataset', 'dataset_selected_feature'))

