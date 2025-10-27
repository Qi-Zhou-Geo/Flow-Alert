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
from functions.warning_strategy.buffer_prediction import cal_buffered_cm

plt.rcParams.update({'font.size': 7,
                     'font.family': "Arial",
                     'axes.formatter.limits': (-4, 6),
                     'axes.formatter.use_mathtext': True})

def convert_df2cm(batch_size, seq_length, DF_threshold, num_repeat, model_version):

    df = pd.read_csv(f"{project_root}/pipeline/{model_version}/test_2022_{num_repeat}repeat"
                     f"/Illgraben-9S-2022-ILL12-EHZ-H-testing-True-{model_version}-H-b{batch_size}-s{seq_length}-{num_repeat}.txt",
                     header=0)

    array_temp = df.values
    time_window_start_float, pre_y_pro = array_temp[:, 0].astype(float), array_temp[:, -2].astype(float)

    obs_y_label = array_temp[:, 2].astype(int)
    pre_y_label = (pre_y_pro >= DF_threshold).astype(int)

    print(f"b{batch_size}-s{seq_length}")
    cm_buffered, f1_buffered = cal_buffered_cm(obs_y_label, pre_y_label)
    tn, fp, fn, tp = cm_buffered.ravel()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = f1_buffered

    return precision, recall, f1

def summary_cm(batch_size_l, seq_length_l, model_version, DF_threshold = 0.5, num_repeat = 9):

    evaluation_precision = np.empty((len(batch_size_l), len(seq_length_l)))
    range_precision = np.empty((len(batch_size_l), len(seq_length_l)), dtype=object)

    evaluation_recall = np.empty((len(batch_size_l), len(seq_length_l)))
    range_recall = np.empty((len(batch_size_l), len(seq_length_l)), dtype=object)

    evaluation_f1 = np.empty((len(batch_size_l), len(seq_length_l)))
    range_f1 = np.empty((len(batch_size_l), len(seq_length_l)), dtype=object)

    for idb, b in enumerate(batch_size_l):
        for ids, s in enumerate(seq_length_l):
            precision, recall, f1 = convert_df2cm(batch_size=b, seq_length=s,
                                                  DF_threshold=DF_threshold,
                                                  num_repeat=num_repeat,
                                                  model_version=model_version)

            evaluation_precision[idb, ids] = precision
            evaluation_recall[idb, ids] = recall
            evaluation_f1[idb, ids] = f1

    evaluation_precision = pd.DataFrame(evaluation_precision, index=batch_size_l, columns=seq_length_l)
    evaluation_recall = pd.DataFrame(evaluation_recall, index=batch_size_l, columns=seq_length_l)
    evaluation_f1 = pd.DataFrame(evaluation_f1, index=batch_size_l, columns=seq_length_l)


    return evaluation_precision, evaluation_recall, evaluation_f1

def plot_cm(evaluation_precision, evaluation_recall, evaluation_f1, model_version, DF_threshold = 0.5):

    purpose_str = "2022 ILL Testing"
    fig = plt.figure(figsize=(4, 7.5))
    gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 1, 0.05])

    for idx, (values, matrix_type) in enumerate(
            zip([evaluation_precision, evaluation_recall, evaluation_f1],
                ["Precision", "Recall", "F1"])):
        ax = plt.subplot(gs[idx])
        heatmap = sns.heatmap(values, vmin=0, vmax=1,
                              annot=True, cmap='coolwarm', fmt=".3f",
                              square=True, cbar=False, ax=ax, annot_kws={"size": 6})
        ax.invert_yaxis()
        ax.set_ylabel("Batch Size", fontweight='bold')
        ax.set_xlabel("Sequence Length", fontweight='bold')
        ax.set_title(f"{purpose_str}-{matrix_type}", fontweight='bold', fontsize=8)

    cbar_ax = plt.subplot(gs[3, :])
    cbar = fig.colorbar(heatmap.collections[0], cax=cbar_ax, orientation="horizontal")
    cbar.set_label(f"Matrix Values", fontsize=6, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{project_root}/pipeline/{model_version}/test_2022_9repeat/heatmap_{model_version}_{purpose_str}_{DF_threshold}.png", dpi=600, transparent=True)
    # plt.show()
    plt.close(fig=fig)


def main(model_version):
    batch_size_l = [64, 128, 256, 512]
    seq_length_l = [16, 32, 48, 64, 96, 128]

    evaluation_precision, evaluation_recall, evaluation_f1 = summary_cm(batch_size_l,
                                                                        seq_length_l,
                                                                        model_version = model_version,
                                                                        DF_threshold = 0.5,
                                                                        num_repeat = 9)

    plot_cm(evaluation_precision, evaluation_recall, evaluation_f1, model_version = model_version, DF_threshold = 0.5)


if __name__ == "__main__":
    import argparse

    # sinfo -n node[501-514] -N --Format="Nodelist,CPUsState,AllocMem,Memory,GresUsed,Gres"
    parser = argparse.ArgumentParser(description='input parameters')
    parser.add_argument("--model_version", type=str)
    args = parser.parse_args()

    main(model_version=args.model_version)
