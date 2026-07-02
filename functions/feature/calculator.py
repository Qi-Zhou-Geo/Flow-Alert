#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2026-05-03
# __author__ = Qi Zhou and Sibashish Dash, GFZ Helmholtz Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission

import numpy as np

# region ### add the sys.path to search for custom modules ###
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
current_dir = current_file.parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent

sys.path.append(str(project_root))
# endregion

# import the custom functions
from functions.feature.feature_name import load_feature_name
from functions.feature.Type_A_features import calBL_feature
from functions.feature.Type_B_features import calculate_all_attributes



def cal_attributes_A(data_array, scaling=1e9, ruler=100): # the main function is from Qi
    data_array_nm = data_array * scaling # converty m/s to nm/s
    # the physical velocity of lower bodunday = 1e-9 m/s and upper boundary as 1e-4
    data_array_nm = np.clip(data_array_nm, a_min=1, a_max=1e5)
    feature_array = calBL_feature(data_array_nm, ruler)

    return feature_array # 17 features


def cal_attributes_B(data_array, sps): # the main function is from Clement
    # sps: sampling frequency; flag=0: one component seismic signal;
    features = calculate_all_attributes(Data=data_array, sps=sps, flag=0)[0] # feature 1 to 60
    feature_array = features[1:]# leave features[0]=event_duration

    return feature_array # 59 features


def feature_calculator(seismic_data, sps, num_meta=4):
    
    feature_Name_A, feature_Name_B = load_feature_name(print_log=False) # return as list
    
    try:
        type_A_arr = cal_attributes_A(data_array=seismic_data)
        type_B_arr = cal_attributes_B(data_array=seismic_data, sps=sps)
    except Exception as e:
        type_A_arr = np.full(shape=len(feature_Name_A) - num_meta, fill_value=np.nan)
        type_B_arr = np.full(shape=len(feature_Name_B)- num_meta, fill_value=np.nan)
        print(f"Warning! len(seismic_data)={len(seismic_data)}, sps={sps}\n"
              f"Error: {e}\n"
              f"Fill gap as: np.nan\n") 
    
    feature_name = np.array(feature_Name_A + feature_Name_B[num_meta:], dtype="object")
    feature_arr = np.append(type_A_arr, type_B_arr)
    
    return feature_name, feature_arr
