# !/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2024-02-23
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission

import yaml
import os
import argparse


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


# <editor-fold desc="add Arial font">
import sys, platform, getpass
# Specify the directory containing the Arial font
if platform.system() == "Linux" and getpass.getuser() == "qizhou":

    from matplotlib import font_manager
    font_dirs = ['/storage/vast-gfz-hpc-01/home/qizhou/2python/font']
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)
# </editor-fold>


# import the custom functions


def clean_data(model_version):

    file_path = (f"{project_root}/pipeline/{model_version}/train_test_2017-2020_9repeat/"
                 f"LSTM/summary_LSTM_optimal.txt")
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

    precision = np.array(df.iloc[:, 19])
    recall = np.array(df.iloc[:, 20])
    f1 = np.array(df.iloc[:, 21])

    evaluation_precision = np.empty((len(batch_size_l), len(seq_length_l)))
    range_precision = np.empty((len(batch_size_l), len(seq_length_l)), dtype=object)

    evaluation_recall = np.empty((len(batch_size_l), len(seq_length_l)))
    range_recall = np.empty((len(batch_size_l), len(seq_length_l)), dtype=object)

    evaluation_f1 = np.empty((len(batch_size_l), len(seq_length_l)))
    range_f1 = np.empty((len(batch_size_l), len(seq_length_l)), dtype=object)

    for idb, b in enumerate(batch_size_l):
        for ids, s in enumerate(seq_length_l):

            idx = np.where((purpose == purpose_str) & (batch_size == f'b{b}') & (seq_length == f's{s}'))[0]
            print(b, s, len(idx))
            print(f1[idx])
            print(df.iloc[idx, 15:19])
            print(np.mean(precision[idx]), np.mean(recall[idx]), np.mean(f1[idx]), "\n")


            evaluation_precision[idb, ids] = np.mean(precision[idx])
            temp = np.max(precision[idx]) - np.min(precision[idx])
            temp = np.round(temp / 2, 3)
            range_precision[idb, ids] = f"{np.mean(precision[idx]):.3f}\n±{temp}"

            evaluation_recall[idb, ids] = np.mean(recall[idx])
            temp = np.max(recall[idx]) - np.min(recall[idx])
            temp = np.round(temp / 2, 3)
            range_recall[idb, ids] = f"{np.mean(recall[idx]):.3f}\n±{temp}"

            evaluation_f1[idb, ids] = np.mean(f1[idx])
            temp = np.max(f1[idx]) - np.min(f1[idx])
            temp = np.round(temp / 2, 3)
            range_f1[idb, ids] = f"{np.mean(f1[idx]):.3f}\n±{temp}"



    evaluation_precision = pd.DataFrame(evaluation_precision, index=batch_size_l, columns=seq_length_l)
    evaluation_recall = pd.DataFrame(evaluation_recall, index=batch_size_l, columns=seq_length_l)
    evaluation_f1 = pd.DataFrame(evaluation_f1, index=batch_size_l, columns=seq_length_l)

    range_precision = pd.DataFrame(range_precision, index=batch_size_l, columns=seq_length_l)
    range_recall = pd.DataFrame(range_recall, index=batch_size_l, columns=seq_length_l)
    range_f1 = pd.DataFrame(range_f1, index=batch_size_l, columns=seq_length_l)


    return evaluation_precision, evaluation_recall, evaluation_f1, range_precision, range_recall, range_f1


def plot_model_results(evaluation_matrix_dict, model_version):

    plt.rcParams.update({'font.size': 7,
                         'font.family': "Arial",
                         'axes.formatter.limits': (-4, 6),
                         'axes.formatter.use_mathtext': True})

    column_map = {"Training":0, "testing":1}
    raw_map = {"Precision":0,"Recall":1, "F1":2}

    fig = plt.figure(figsize=(6, 7.5))
    gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 1, 0.05])

    for keys in evaluation_matrix_dict.keys():
        purpose_str, matrix_type = keys.split("-")
        values_temp = evaluation_matrix_dict[keys]
        values,  annot = values_temp
        print(annot)
        ax = plt.subplot(gs[
                             int(raw_map.get(matrix_type)),
                             int(column_map.get(purpose_str)),

                         ])

        heatmap = sns.heatmap(values, vmin=0, vmax=1,
                              annot=annot, cmap='coolwarm', fmt="",
                              square=True, cbar=False, ax=ax, annot_kws={"size": 6, "color": "black"})
        ax.invert_yaxis()
        ax.set_ylabel("Batch Size", fontweight='bold')
        ax.set_xlabel("Sequence Length", fontweight='bold')
        ax.set_title(f"{purpose_str.capitalize()}-{matrix_type}", fontweight='bold', fontsize=8)

    cbar_ax = plt.subplot(gs[3, :])
    cbar = fig.colorbar(heatmap.collections[0], cax=cbar_ax, orientation="horizontal")
    cbar.set_label(f"Matrix Values", fontsize=6, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{project_root}/pipeline/{model_version}/train_test_2017-2020_9repeat/heatmap_{model_version}_train_test.png", dpi=600, transparent=True)
    plt.close(fig=fig)

def main(model_version):
    df = clean_data(model_version)

    temp_dict = {}
    for purpose_str in ["Training", "testing"]:
        evaluation_matrix= create_evaluation_matrix(df, purpose_str,
                                                   batch_size_l=(64, 128, 256, 512),
                                                   seq_length_l=(16, 32, 48, 64, 96, 128))
        for idx, matrix_type in enumerate(["Precision", "Recall", "F1"]):

            keys = f"{purpose_str}-{matrix_type}"
            values1 = evaluation_matrix[idx]
            values2 = evaluation_matrix[idx+3]
            temp_dict[keys] = (values1, values2)
        print(purpose_str)

    plot_model_results(evaluation_matrix_dict=temp_dict, model_version=model_version)

if __name__ == "__main__":
    # sinfo -n node[501-514] -N --Format="Nodelist,CPUsState,AllocMem,Memory,GresUsed,Gres"
    parser = argparse.ArgumentParser(description='input parameters')
    parser.add_argument("--model_version", type=str)
    args = parser.parse_args()

    main(model_version=args.model_version)