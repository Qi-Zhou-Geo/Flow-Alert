#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2025-08-05
# __author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission

import os
import sys
import yaml

import numpy as np
import pandas as pd

from tqdm import tqdm

from datetime import datetime, timezone, date, timedelta

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec


# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path
current_dir = Path(__file__).resolve().parent
# using ".parent" on "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys
sys.path.append(str(project_root))
# </editor-fold>

# import the custom functions
from functions.data_process.load_data import select_features, clip_df_columns

plt.rcParams.update( {'font.size':7,
                      'font.family': "Arial",
                      'axes.formatter.limits': (-8, 6),
                      'axes.formatter.use_mathtext': True} )


catchment_name = "Illgraben"
feature_type = "D"
station = 2 # 2 -> IGB/ILL 02 or 12 station
param = [
f"Illgraben-9J-2013-IGB0*-HHZ-{feature_type}-training-False",
f"Illgraben-9J-2014-IGB0*-HHZ-{feature_type}-training-False",
f"Illgraben-9S-2017-ILL0*-EHZ-{feature_type}-training-True",
f"Illgraben-9S-2018-ILL1*-EHZ-{feature_type}-training-True",
f"Illgraben-9S-2019-ILL1*-EHZ-{feature_type}-training-True",
]


for i, p in enumerate(param):

    p = p.replace("*", f"{station}")
    catchment_name, seismic_network, input_year, input_station, input_component, \
    feature_type, dataloader_type, with_label = p.split("-")

    input_features_name, data_array = select_features(catchment_name,
                                                      seismic_network,
                                                      input_year,
                                                      input_station,
                                                      input_component,
                                                      feature_type,
                                                      with_label=False,
                                                      repeat=1,
                                                      normalize=False)
    df = pd.DataFrame(data_array)

    if i == 0:
        df_all = df
    else:
        # concatenating df as new rows at the bottom of df_all
        df_all = pd.concat([df_all, df], axis=0, ignore_index=True)

network_features = pd.DataFrame(np.random.rand(df_all.shape[0], 10)) # 10 network featrues
network_features.columns = ['id_maxRMS', 'id_minRMS', 'ration_maxTOminRMS', 'ration_maxTOminIQR', 'mean_coherenceOfNet',
                            'max_coherenceOfNet', 'mean_lagTimeOfNet', 'std_lagTimeOfNet', 'mean_wdOfNet',
                            'std_wdOfNet']

df_all = pd.concat([df_all.iloc[:, :-1], network_features, df_all.iloc[:, -1]], axis=1)

df_all.to_csv(f"{project_root}/data/scaler/2017-2022-02station-{catchment_name}-{feature_type}.txt", sep=',',
              index=False, index_label=False)

df_features = df_all.iloc[:, 1:-1]
print(df_features.columns, df_features.shape)
df_features = clip_df_columns(df_features)
min_vals = df_features.min()
max_vals = df_features.max()
df_normalized = (df_features - min_vals) / (max_vals - min_vals)

# 2013-2014, 2017-2020, 2020 ILL/IGB
note = f"This normalize factors are basde on 2017-2019 02/12 station, " \
       f"the used feature_type is {feature_type}, processed time is {datetime.now()}"
np.savez(
    f"{project_root}/data/scaler/normalize_factor4{feature_type}.npz",
    min_factor=min_vals.values.reshape(1, -1),
    max_factor=max_vals.values.reshape(1, -1),
    note=note
)