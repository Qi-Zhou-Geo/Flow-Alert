# !/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2024-02-23
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission

import os
import argparse

from datetime import datetime

# print("PyTorch version:", torch.__version__) = PyTorch version: 1.12.1
import numpy as np
import pandas as pd
import torch.optim as optim
from torchinfo import summary
# print("Torchinfo version:", torchinfo.__version__) = Torchinfo version: 1.8.0


# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path

current_dir = Path(__file__).resolve().parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent.parent
import sys

sys.path.append(str(project_root))
# </editor-fold>


# import the custom functions
from functions.public.load_data import select_features

station_list = ["STA01", "STA02", "STA03", "STA04", "STA05", "STA07", "STA08"]

for station in station_list:
    df = pd.read_csv(f"{project_root}/demo/Yanmen/min_max/{station}.txt", header=None)
    df_arr = np.array(df)
    date = df_arr[:, 0]
    feature = df_arr[:, 1:]


    clipped = np.empty_like(feature)
    for i in range(feature.shape[1]):
        lower = np.percentile(feature[:, i], 1)
        upper = np.percentile(feature[:, i], 99)
        clipped[:, i] = np.clip(feature[:, i], lower, upper)

    min_arr = np.min(clipped, axis=0)
    mean_arr = np.mean(clipped, axis=0)
    max_arr = np.max(clipped, axis=0)
    np.savez(f"{project_root}/demo/Yanmen/min_max/{station}_vlaues.npz", min_arr=min_arr, max_arr=max_arr)


    clipped = np.hstack((date.reshape(-1, 1), clipped))
    np.savetxt(f"{project_root}/demo/Yanmen/min_max/{station}_new.txt", X=clipped, delimiter=',', fmt="%s")

