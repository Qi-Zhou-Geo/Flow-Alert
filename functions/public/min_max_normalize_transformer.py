#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-02-23
#__author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
#__find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission

import numpy as np
import pandas as pd
from joblib import dump, load

# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path
current_dir = Path(__file__).resolve().parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys
sys.path.append(str(project_root))
# </editor-fold>

# import the custom functions
from functions.public.min_max_normalize_fit import get_scaler_name


def min_max_normalize(x, input_station, feature_type,
                      scaler_out_dir=None):
    '''
    Normalize the train and test data-60s by MinMaxScaler()

    Args:
        x: pandas dataframe, shape by (num_time_stamps, num_feature)
        scaler_name: string, f"min_max_scaler_2017_2019_{feature_type}"
    Returns:
        x: normalized pandas dataframe, shape by (num_time_stamps, num_feature)

    '''
    scaler_name = get_scaler_name(input_station, feature_type)

    if scaler_out_dir is None:
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parent.parent
        scaler_out_dir = f"{project_root}/functions/public/scaler"

    scaler = load(f"{scaler_out_dir}/{scaler_name}.joblib")

    x_numeric = np.array(x, dtype=float)
    x_scaled = scaler.transform(x_numeric) # x without feature name


    return x_scaled
