#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2025-05-30
# __author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission

import os
import argparse

import yaml

import torch
import numpy as np
import pandas as pd

from tqdm import tqdm

from datetime import datetime, timezone, timedelta

# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path
current_dir = Path(__file__).resolve().parent
# using ".parent" on "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys
sys.path.append(str(project_root))
# </editor-fold>

# import the custom functions
from functions.data_process.cross_catchments_inference import load_st, load_model, load_feature_as_dataLoadter
from functions.data_process.cross_catchments_inference import make_prediction_from_dataLoader, find_max_amp_time
from functions.data_process.cross_catchments_inference import plot_predicted_pro, _dump_results

from functions.warning_strategy.calculate_inference_matrix import inference_matrix
from functions.toolkit.multi_process_archive import dump_as_row
from functions.warning_strategy.buffer_prediction import cal_buffered_cm
from functions.visualize.heatmap_plot import visualize_probability_map


def plot_pro(time_window_start_float, pre_y_pro, output_path, output_format):

    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(5.5, 5))
    gs = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs[0])

    visualize_probability_map(ax, time_window_start_float, pre_y_pro)

    plt.tight_layout()
    plt.savefig(f"{output_path}/{output_format}.png", dpi=600, transparent=True)
    plt.show()
    plt.close(fig=fig)

def main(params,
         model_version,
         num_repeat,
         feature_type,
         batch_size,
         seq_length,
         output_path=None):

    # st, params, sta_s, sta_e = load_st(idx, buffer=24)
    output_format = [params, model_version, feature_type,
                     f"b{batch_size}", f"s{seq_length}", num_repeat]

    output_format = [str(i) for i in output_format]
    output_format = "-".join(output_format)

    # model
    ensemble_pre_trained_LSTM, models = load_model(model_version, feature_type, batch_size, seq_length, num_repeat)

    # dataloader
    sub_window_size = 60
    synthetic = False
    dataLoader = load_feature_as_dataLoadter(params, batch_size, sub_window_size, seq_length, synthetic=synthetic)

    # model prediction
    # ["t_target", "t_str", "label", "pro-i", ..., "pro_mean", "ci_range"]
    array_temp = make_prediction_from_dataLoader(ensemble_pre_trained_LSTM, models, dataLoader)

    # if synthetic is True:
    #     array_temp = array_temp[synthetic_length * seq_length :, :]  # clip back to the origional length

    # dump the results
    if output_path is None:
        output_path = f"{current_dir}/output"
    else:
        output_path = output_path
    os.makedirs(output_path, exist_ok=True)
    sta_s, sta_e = None, None
    benchmark_time = None
    _dump_results(sta_s, sta_e, benchmark_time, array_temp, models, output_format, output_path)

    time_window_start_float, pre_y_pro = array_temp[:, 0].astype(float), array_temp[:, -2].astype(float)

    obs_y_label = array_temp[:, 2].astype(int)
    pre_y_label = (pre_y_pro >= 0.5).astype(int)

    cal_buffered_cm(obs_y_label, pre_y_label)
    plot_pro(time_window_start_float, pre_y_pro, output_path, output_format)

if __name__ == "__main__":
    # sinfo -n node[501-514] -N --Format="Nodelist,CPUsState,AllocMem,Memory,GresUsed,Gres"
    parser = argparse.ArgumentParser(description='input parameters')
    # parser.add_argument("--params_list", type=str, help="parameters-string")
    parser.add_argument("--params", type=str, help="parameters-string")
    parser.add_argument("--model_version", type=str)
    parser.add_argument("--num_repeat", type=int)
    parser.add_argument("--feature_type", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--seq_length", type=int)
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()

    # params_list = args.params_list.split()
    # for params in params_list:
    #     main(params, args.model_version, args.num_repeat, args.feature_type, args.batch_size, args.seq_length)
    main(args.params, args.model_version, args.num_repeat,
         args.feature_type, args.batch_size, args.seq_length, args.output_path)
