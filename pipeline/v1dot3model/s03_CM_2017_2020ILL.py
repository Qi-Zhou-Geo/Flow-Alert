#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2025-12-14
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission

import os
import argparse

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import seaborn as sns

# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path

current_dir = Path(__file__).resolve().parent

# using ".parent" on "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys

sys.path.append(str(project_root))
# </editor-fold>


# import the custom functions

plt.rcParams.update({'font.size': 7,
                     'axes.formatter.limits': (-4, 6),
                     'axes.formatter.use_mathtext': True})
model_version = "v1dot3model"
batch_size = "b128"
seq_length = "s64"
num_repeat = 9



df = pd.read_csv(f"{project_root}/pipeline/{model_version}/train_test_2017-2020_9repeat/LSTM/summary_LSTM_optimal.txt",
                 header=None)
df_arr = np.array(df)

fig = plt.figure(figsize=(6, 9))
gs = gridspec.GridSpec(9, 2)

for purpose in ["Training", "testing"]:
    for repeat in range(num_repeat):

        if purpose == "Training":
            column = 0
        else:
            column = 1

        index = np.where( (df_arr[:, 0] == purpose) &
                         (df_arr[:, 13] == batch_size) &
                         (df_arr[:, 14] == seq_length) &
                         (df_arr[:, 10] == int(repeat+1)) )[0][0]

        cm = np.array(df_arr[index, 15:19]).reshape(2, 2).astype(float)
        f1 = df_arr[index, 21]

        ax = plt.subplot(gs[repeat, column])
        sns.heatmap(cm, ax=ax, cbar=False, annot=True, fmt='.0f')
        ax.set_title(label=f"{purpose.capitalize()}, Initialization={repeat+1}, F1={f1}",
                     fontsize=7, loc="left", fontweight='bold')

        if purpose == "Training":
            ax.set_ylabel("Labeled", fontweight='bold')
            ax.set_yticks([0.5, 1.5], ["Non-DF", "DF"])
        else:
            ax.set_ylabel("", fontweight='bold')
            ax.set_yticks([0.5, 1.5], ["", ""])

        if repeat == 8:
            ax.set_xlabel("Predicted", fontweight='bold')
            ax.set_xticks([0.5, 1.5], ["Non-DF", "DF"])
        else:
            ax.set_xticks([0.5, 1.5], ["", ""])


plt.tight_layout()
plt.savefig(f"{project_root}/pipeline/{model_version}/train_test_2017-2020_9repeat/cm_{batch_size}_{seq_length}_training_testing.png",
            dpi=600)  # , transparent=True
plt.show()
plt.close(fig=fig)
