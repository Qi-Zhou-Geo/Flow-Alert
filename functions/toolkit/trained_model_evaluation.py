# !/usr/bin/python
# -*- coding: UTF-8 -*-
import pandas as pd
# __modification time__ = 2024-02-23
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission

import yaml
import os

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec


from obspy import read, Trace, Stream, read_inventory, signal
from obspy.core import UTCDateTime

# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path
current_dir = Path(__file__).resolve().parent
# using ".parent" on "pathlib.Path" object moves one level up the directory hierarchy

project_root = current_dir.parent.parent
import sys
sys.path.append(str(project_root))
# </editor-fold>


# import the custom functions
from functions.data_process.load_data import select_features
from functions.seismic.seismic_data_processing import load_seismic_signal

def clean_data(model_version):

    file_path = f"{project_root}/pipeline/{model_version}/LSTM/summary_LSTM_optimal_{model_version[:3]}.txt"
    with open(file_path, "r") as f:
        text = f.read()

    text = text.replace("[", "").replace("]", "").replace("-", ",")
    with open(file_path, "w") as f:
        f.write(text)

    df = pd.read_csv(file_path, header = None)

    return df


def create_evaluation_matrix(df,
                             purpose_str='testing',
                             batch_size_l=(64, 128, 256, 512),
                             seq_length_l=(16, 32, 64, 128)):

    purpose = np.array(df.iloc[:, 0])
    repeat = np.array(df.iloc[:, 10])
    batch_size = np.array(df.iloc[:, 13])
    seq_length = np.array(df.iloc[:, 14])
    f1 = np.array(df.iloc[:, 21])

    evaluation_matrix = np.empty((len(batch_size_l), len(seq_length_l)))

    for idb, b in enumerate(batch_size_l):
        for ids, s in enumerate(seq_length_l):
            idx = np.where((purpose == purpose_str) & (batch_size == f'b{b}') & (seq_length == f's{s}'))[0]
            print(b, s)
            print(repeat[idx])
            print(f1[idx])
            print(df.iloc[idx, 15:19])
            print(np.mean(f1[idx]), "\n")

            evaluation_matrix[idb, ids] = np.mean(f1[idx])

    evaluation_matrix = pd.DataFrame(evaluation_matrix, index=seq_length_l, columns=batch_size_l)

    return evaluation_matrix

def plot_heatmap(evaluation_matrix, purpose_str='testing'):

    plt.rcParams.update({'font.size': 7,
                         'font.family': "Arial",
                         'axes.formatter.limits': (-4, 6),
                         'axes.formatter.use_mathtext': True})



    fig = plt.figure(figsize=(5, 5))
    gs = gridspec.GridSpec(1, 1)

    ax = plt.subplot(gs[0])
    heatmap = sns.heatmap(evaluation_matrix, vmin=0, vmax=1,
                          annot=True, fmt=".3g", cmap='coolwarm',
                          square=True, cbar=True, ax=ax)
    ax.invert_yaxis()
    ax.set_xlabel("Batch Size", fontweight='bold')
    ax.set_ylabel("Sequence Length", fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{project_root}/pipeline/v35model/heatmap_{purpose_str}.png", dpi=600, transparent=True)
    plt.show()
    plt.close(fig=fig)

model_version = "v35model"
purpose_str = "testing" #"training" #"testing"
df = clean_data(model_version)

evaluation_matrix = create_evaluation_matrix(df, purpose_str)
plot_heatmap(evaluation_matrix, purpose_str)