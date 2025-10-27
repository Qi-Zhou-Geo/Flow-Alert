# !/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2024-02-23
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission

import yaml
import os
import argparse

from scipy.stats import goodness_of_fit
from tqdm import tqdm
from datetime import datetime

import matplotlib.pyplot as plt

# print("PyTorch version:", torch.__version__) = PyTorch version: 1.12.1
import numpy as np
import pandas as pd
import torch.optim as optim
# from torchinfo import summary
# print("Torchinfo version:", torchinfo.__version__) = Torchinfo version: 1.8.0

from obspy import read, Trace, Stream, read_inventory, signal
from obspy.core import UTCDateTime

import seaborn as sns
import matplotlib.cm as cm
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
from calculate_features.Type_A_features import calBL_feature
from functions.seismic.chunk_st2seq import chunk_data

plt.rcParams.update({'font.size': 7,
                     'font.family': "Arial",
                     'axes.formatter.limits': (-4, 6),
                     'axes.formatter.use_mathtext': True})

def cal_attributes_A(data_array, scaling=1e9, ruler=100): # the main function is from Qi

    data_array_nm = data_array * scaling # converty m/s to nm/s
    # the physical velocity of lower bodunday = 1e-9 m/s and upper boundary as 1e-4
    data_array_nm = np.clip(data_array_nm, a_min=1, a_max=1e5)
    feature_array = calBL_feature(data_array_nm, ruler)

    return feature_array # 17 features


st = read(f"{project_root}/demo/Jiangjia/JJG.453007897.EHZ.2025.199.disp.mseed")
st.plot()
print(st[0].stats)
st[0].stats.sampling_rate = 250
print(st[0].stats)

st_data = st[0].data
sub_window_size, window_overlap = 60, 0
st_startime_float, st_endtime_float = float(st[0].stats.starttime), float(st[0].stats.endtime)
st_sps, npts = st[0].stats.sampling_rate, st[0].stats.npts
chunk_t_str, chunk_t, chunk_x = chunk_data(st_data,
                                           sub_window_size, window_overlap,
                                           st_startime_float, st_endtime_float,
                                           st_sps, npts)

# random select time step to get the feature shape
temp = cal_attributes_A(data_array=chunk_x[100], scaling=1e9, ruler=100)
bl_feature = np.empty((chunk_t_str.shape[0], temp.shape[0] + 2), dtype=object)

for i, (t_str, t_float, x) in enumerate(tqdm(zip(chunk_t_str, chunk_t, chunk_x)),
                                        total=len(chunk_x)):
    features = cal_attributes_A(data_array=x, scaling=1e9, ruler=100)
    temp = [t_str, t_float] + features.tolist()

    bl_feature[i] = np.array(temp, dtype=object)

# this index  may differ than the workflow, please double check
goodness_of_fit = 12
alpha = 15

plt.plot(bl_feature[:, goodness_of_fit].astype(float), label="goodness-of-fit")
plt.ylim(-100, 100)
plt.show()
plt.plot(bl_feature[:, alpha].astype(float), label="alpha")
plt.ylim(0, 3)
plt.show()
