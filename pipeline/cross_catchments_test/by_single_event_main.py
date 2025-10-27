#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2025-05-30
# __author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission

import os
import argparse

import yaml

import pickle
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm

# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path

current_dir = Path(__file__).resolve().parent
# using ".parent" on "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys

sys.path.append(str(project_root))
# </editor-fold>

# import the custom functions
from functions.data_process.cross_catchments_inference import load_st, load_model, load_feature_as_sequences
from functions.data_process.cross_catchments_inference import make_prediction_from_sequences, find_max_amp_time
from functions.data_process.cross_catchments_inference import plot_predicted_pro, _dump_results

from functions.warning_strategy.calculate_inference_matrix import inference_matrix
from functions.toolkit.multi_process_archive import dump_as_row


def main(idx,
         model_version,
         num_repeat,
         feature_type,
         batch_size, seq_length,
         sub_window_size, window_overlap,
         synthetic_length,
         output_path=None):

    st, params, sta_s, sta_e = load_st(idx=idx, buffer=24, f_min=1, f_max=25)
    benchmark_time = find_max_amp_time(st, sta_s, sta_e)

    output_format = [str(idx + 1).zfill(3), params, model_version, feature_type,
                     f"b={batch_size}", f"s={seq_length}", num_repeat]

    output_format = [str(i) for i in output_format]
    output_format = "-".join(output_format)
    # set output path
    if output_path is None:
        output_path = f"{current_dir}/output"
    else:
        output_path = output_path
    os.makedirs(output_path, exist_ok=True)

    # load model
    ensemble_pre_trained_LSTM, models = load_model(model_version, feature_type, batch_size, seq_length, num_repeat)

    # load the sequence, this will save the time
    cached_sequences = Path(f"{output_path}/{output_format}_cached_sequences.pkl")
    if cached_sequences.exists():
        with open(cached_sequences, "rb") as f:
            sequences = pickle.load(f)
    else:
        sequences = load_feature_as_sequences(st, sub_window_size, window_overlap, feature_type,
                                              seq_length, synthetic_length)
        with open(cached_sequences, "wb") as f:
            pickle.dump(sequences, f)

    # array_temp as ["t_target", "t_str", "label", "pro-i", ..., "pro_mean", "ci_range"]
    array_temp = make_prediction_from_sequences(ensemble_pre_trained_LSTM, models, sequences)
    _dump_results(sta_s, sta_e, benchmark_time, array_temp, models, output_format, output_path)


    temp = inference_matrix(array_temp, benchmark_time, sta_s, sta_e, pro_epsilon=0.5, buffer1=3, buffer2=3)
    detection_type, first_detection, first_detection_str, increased_warning_time, false_detection, false_detection_ratio = temp

    # dump the results
    output_dir = output_path
    output_name = "inference-Non-ILL-event"
    variable_str = output_format
    record = [detection_type, first_detection_str, benchmark_time, increased_warning_time, false_detection, false_detection_ratio]
    dump_as_row(output_dir, output_name, variable_str, *record)

    # plot
    note = (f"detection_type={detection_type},\n"
            f"benchmark_time={benchmark_time},\n"
            f"first_detection_str={first_detection_str}, \n"
            f"increased_warning_time={increased_warning_time} [seconds], \n"
            f"false_detection_ratio={false_detection_ratio}, \n")
    plot_predicted_pro(benchmark_time, first_detection_str, st, array_temp, output_path, output_format, note)


if __name__ == "__main__":
    # sinfo -n node[501-514] -N --Format="Nodelist,CPUsState,AllocMem,Memory,GresUsed,Gres"
    parser = argparse.ArgumentParser(description='input parameters')
    parser.add_argument("--idx", type=int, help="event index in catalog (from 1)")
    parser.add_argument("--model_version", type=str)
    parser.add_argument("--num_repeat", type=int)
    parser.add_argument("--feature_type", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--seq_length", type=int)
    parser.add_argument("--sub_window_size", type=int)
    parser.add_argument("--window_overlap", type=int)
    parser.add_argument("--synthetic_length", type=int)
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()

    idx = args.idx - 1

    main(idx, args.model_version, args.num_repeat, args.feature_type, args.batch_size, args.seq_length,
         args.sub_window_size, args.window_overlap, args.synthetic_length, args.output_path)
