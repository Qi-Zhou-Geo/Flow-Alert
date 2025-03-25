#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-02-23
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission

import numpy as np
import pandas as pd
from joblib import dump, load


# <editor-fold desc="add the sys.path">
import sys, platform
if platform.system() == 'Darwin': # add the parent_dir to the sys
    sys.path.append("/Users/qizhou/#python/#GitHub_saved/Diversity-of-Debris-Flow-Footprints")
elif platform.system() == 'Linux':
    sys.path.append('/home/qizhou/3paper/3Diversity-of-Debris-Flow-Footprints')
else:
    print(f"please add 'sys.path' for your platform.system() == {platform.system()}")
# </editor-fold>

# import the custom functions
from config.config_dir import CONFIG_dir
from functions.public.min_max_normalize_fit import get_scaler_name


def min_max_normalize(x, input_station, feature_type,
                      scaler_out_dir=f"{CONFIG_dir['parent_dir']}/functions/public/scaler"):
    '''
    Normalize the train and test data by MinMaxScaler()

    Args:
        x: pandas dataframe, shape by (num_time_stamps, num_feature)
        scaler_name: string, f"min_max_scaler_2017_2019_{feature_type}"
    Returns:
        x: normalized pandas dataframe, shape by (num_time_stamps, num_feature)

    '''

    scaler_name = get_scaler_name(input_station, feature_type)
    scaler = load(f"{scaler_out_dir}/{scaler_name}.joblib")

    x_numeric = np.array(x, dtype=float)
    x_scaled = scaler.transform(x_numeric) # x without feature name


    return x_scaled
