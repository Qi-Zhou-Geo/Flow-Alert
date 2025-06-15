#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-02-23
#__author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
#__find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
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
from functions.public.load_data import select_features

def get_scaler_name(input_station, feature_type):

    station_code_mapping = {
        "ILL02": 2, "ILL08": 8, "ILL03": 3,
        "ILL12": 2, "ILL18": 8, "ILL13": 3}

    scaler_name = f"min_max_scaler_2017_2019_{station_code_mapping.get(input_station)}_{feature_type}"

    return scaler_name


def fit_min_max_scaler(x, input_station, feature_type,
                       scaler_out_dir=None):
    '''
    Fit the MinMaxScaler()

    Args:
        x: pandas dataframe, shape by (num_time_stamps, num_feature)
        scaler_name: string, f"min_max_scaler_2017_2019_{feature_type}"
    Returns:
        no returns
    '''

    # Only import this when you need it,
    # otherwise, you will see (most likely due to a circular import)

    scaler_name = get_scaler_name(input_station, feature_type)
    print(f"{scaler_name}, {input_station}, {feature_type}, {x.shape}")
    scaler = MinMaxScaler()

    x_numeric = np.array(x, dtype=float)
    scaler.fit(x_numeric) # x without feature name

    if scaler_out_dir is None:
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parent.parent
        scaler_out_dir = f"{project_root}/functions/public/scaler"

        os.makedirs(scaler_out_dir, exist_ok=True)
    dump(scaler, f"{scaler_out_dir}/{scaler_name}.joblib")

def fit_scaler(params):

    for feature_type in ["A", "B", "C", "D", "E"]:
        df = pd.DataFrame()
        for p in params:

            seismic_network, input_year, input_station, input_component, dataloader_type, with_label = p.split("-")

            # convert str to bool
            if with_label == "True":
                with_label = True
            else:
                with_label = False

            # load data-60s as data-60s frame
            input_features_name, data_array = select_features(seismic_network,
                                                              input_year,
                                                              input_station,
                                                              input_component,
                                                              feature_type,
                                                              with_label,
                                                              normalize=False)

            # the data-60s array structured as [time_stamp_float, x, y]
            temp = pd.DataFrame(data_array[:, 1:-1], columns=input_features_name)
            df = pd.concat([df, temp], axis=0, ignore_index=True)

        fit_min_max_scaler(df, input_station, feature_type)


def main():
    params = [
        "9S-2017-ILL02-EHZ-training-True",
        "9S-2018-ILL12-EHZ-training-True",
        "9S-2019-ILL12-EHZ-training-True"
    ]
    fit_scaler(params)

    params = [
        "9S-2017-ILL08-EHZ-training-True",
        "9S-2018-ILL18-EHZ-training-True",
        "9S-2019-ILL18-EHZ-training-True"
    ]
    fit_scaler(params)

    params = [
        "9S-2017-ILL03-EHZ-training-True",
        "9S-2018-ILL13-EHZ-training-True",
        "9S-2019-ILL13-EHZ-training-True"
    ]
    fit_scaler(params)


if __name__ == "__main__":
    main()