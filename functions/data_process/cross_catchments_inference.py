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

from obspy import read, UTCDateTime

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
from functions.model.lstm_model import Ensemble_Trained_LSTM_Classifier
from functions.data_process.dataset_to_dataloader import data_to_seq, seq_to_dataset, dataset_to_dataloader
from functions.data_process.df_to_dataloader import prepare_sequences
# for st
from functions.seismic.seismic_data_processing import load_seismic_signal, load_seismic_pieces
from functions.data_process.prepare_feature4inference import Stream_to_feature
from functions.toolkit.convert_time import clean_time
from functions.seismic.st2tr import stream_to_trace
from functions.seismic.plot_obspy_st import rewrite_x_ticks


def _dump_results(sta_s, sta_e, benchmark_time, array_temp, models, output_format, output_path):
    column_name = ([f"t_target={sta_s}", f"t_str={sta_e}", f"label={benchmark_time}"] +
                   [f"pro{i}" for i in range(len(models))] +
                   ["pro_mean", "ci_range"])
    df = pd.DataFrame(array_temp, columns=column_name)

    os.makedirs(output_path, exist_ok=True)
    df.to_csv(f"{output_path}/{output_format}.txt", sep=",", index=False)

    print(f"Dumped {output_format} to {output_path}")


def load_model(model_version, feature_type, batch_size, seq_length, num_repeat):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    ensemble_pre_trained_LSTM = Ensemble_Trained_LSTM_Classifier(model_version,
                                                                 feature_type,
                                                                 batch_size, seq_length,
                                                                 device,
                                                                 ML_name="LSTM",
                                                                 station="ILL02")

    models = ensemble_pre_trained_LSTM.ensemble_models(num_repeat=num_repeat,
                                                       attention=True,
                                                       print_model_summary=True)

    return ensemble_pre_trained_LSTM, models


def load_st(idx, buffer=24, f_min=1, f_max=25):
    '''
    Load the st

    Args:
        idx: int, event from 0-N
        buffer: load signal buffer, unit by hour
        f_min: int or float, unit by Hz
        f_max: int or float, unit by Hz

    Returns:

    '''
    # <editor-fold desc="prepare data">
    default_data_path = f"{project_root}/config/data_path.yaml"
    with open(default_data_path, "r") as f:
        config = yaml.safe_load(f)
        sac_path = config[f"glic_sac_dir"]
        event_catalog_version = config[f"event_catalog_version"]
        print(f"event_catalog_version: {event_catalog_version}")

    file_path = f"{project_root}/data/event_catalog/{event_catalog_version}"
    df = pd.read_csv(f"{file_path}", header=0)

    row_idx = df.loc[idx]  # select row_idx
    print(df, "\n", row_idx)
    continent = row_idx["Continent"]
    catchment = row_idx["Catchment"]
    longitude = row_idx["Longitude-Station(-denote-West)"]
    latitude = row_idx["Latitude-Station(-denote-South)"]
    client = row_idx["Client"]
    seismic_network = row_idx["Network"]
    station = row_idx["Station"]
    location = row_idx["Location"]
    component = row_idx["Component"]
    sps = row_idx["SPS(Hz)"]
    distance = row_idx["Min-Distance2DF-Channel(km)"]
    type_source = row_idx["Type(debris-flow=DF)"]

    data_start = row_idx["Manually-Start-time(UTC+0)"]
    data_end = row_idx["Manually-End-time(UTC+0)"]

    ref4sta_s = row_idx["Ref-Start-time4STA(UTC+0)"]
    ref4sta_e = row_idx["Ref-End-time4STA(UTC+0)"]

    sta_s = row_idx["Start-time(UTC+0)-by-STA/LTA"]
    sta_e = row_idx["End-time(UTC+0)-by-STA/LTA"]
    # </editor-fold>

    # select 24 hours before and 24 hours after the event
    dt1_iso_str, dt2_iso_str = clean_time(time_str1=data_start, time_str2=data_end, buffer=buffer)

    try:
        st = load_seismic_signal(catchment, seismic_network,
                                 station, component,
                                 dt1_iso_str, dt2_iso_str,
                                 f_min=f_min, f_max=f_max,
                                 remove_sensor_response=True,
                                 raw_data=False)
    except (FileNotFoundError, OSError, IOError) as e:
        print(f"Error 1!\n {e}")
        st = load_seismic_pieces(catchment, seismic_network,
                                 station, component,
                                 dt1_iso_str, dt2_iso_str,
                                 f_min=f_min, f_max=f_max,
                                 remove_sensor_response=True,
                                 raw_data=False)
    except Exception as e:
        print(f"Error 2\n {e}")

    output_format = f"{continent}-{catchment}-{seismic_network}-{data_start[:4]}-{station}-{component}"

    return st, output_format, sta_s, sta_e


def find_max_amp_time(st, sta_s, sta_e):
    tr = st.copy()
    tr.trim(UTCDateTime(sta_s), UTCDateTime(sta_e))
    tr = stream_to_trace(st=tr)
    data = tr.data

    index = np.argmax(data)
    benchmark_time = UTCDateTime(tr.stats.starttime) + index * tr.stats.delta
    benchmark_time = benchmark_time.strftime(f"%Y-%m-%dT%H:%M:%S")

    return benchmark_time


def synthetic_input4model(data_array, sub_window_size, seq_length, synthetic_length):
    num_synthetic = synthetic_length * seq_length

    data_beginning = data_array[0, :]  # 1D ([time stamps, features, labels])
    t0 = data_beginning[0]
    f0 = data_beginning[1:]
    synthetic_t0 = np.arange(t0 - num_synthetic * sub_window_size, t0, sub_window_size)
    synthetic_f0 = np.repeat(a=[f0], repeats=num_synthetic, axis=0)
    synthetic_data0 = np.hstack((synthetic_t0.reshape(-1, 1), synthetic_f0))

    data_ending = data_array[-1, :]  # 1D ([time stamps, features, labels])
    t1 = data_ending[0]
    f1 = data_ending[1:]
    synthetic_t1 = np.arange(t1, t1 + sub_window_size * num_synthetic, sub_window_size)
    synthetic_f1 = np.repeat(a=[f1], repeats=num_synthetic, axis=0)
    synthetic_data1 = np.hstack((synthetic_t1.reshape(-1, 1), synthetic_f1))

    synthetic_data_array = np.vstack((synthetic_data0, data_array, synthetic_data1))

    return synthetic_data_array


def load_feature_as_dataLoadter(params, batch_size, sub_window_size, seq_length, synthetic=True):
    catchment_name, seismic_network, input_year, input_station, input_component, \
        feature_type, dataloader_type, with_label = params.split("-")

    if with_label == "True":
        with_label = True

    input_features_name, data_array = select_features(catchment_name,
                                                      seismic_network,
                                                      input_year,
                                                      input_station,
                                                      input_component,
                                                      feature_type,
                                                      with_label,
                                                      normalize=True)
    print(data_array.shape)
    if synthetic is True:
        synthetic_length = 3  # hour
        data_array = synthetic_input4model(data_array, sub_window_size, seq_length, synthetic_length=synthetic_length)

    sequences = data_to_seq(array=data_array, seq_length=seq_length)
    print(len(sequences))
    dataset = seq_to_dataset(sequences=sequences, data_type="feature")
    dataloader = dataset_to_dataloader(dataset=dataset,
                                       batch_size=batch_size,
                                       training_or_testing="testing")
    dataLoader = dataloader.dataLoader()

    # return dataLoader, synthetic_length
    return dataLoader


def load_feature_as_sequences(st, sub_window_size, window_overlap, feature_type, seq_length, synthetic_length):
    stream_to_feature = Stream_to_feature(sub_window_size, window_overlap, feature_type)
    output_feature = stream_to_feature.prepare_feature(st=st)
    feature_arr = output_feature[:, :-3]
    t_str, t_float, pretend_label = output_feature[:, -3], output_feature[:, -2], output_feature[:, -1]

    t_float = t_float.astype(float).reshape(-1, 1)
    pretend_label = pretend_label.astype(float).reshape(-1, 1)

    # stack as column for 2D array
    data_array = np.concatenate((t_float, feature_arr, pretend_label), axis=1)  # 2D ([time stamps, features, labels])
    if synthetic_length > 0:
        data_array = synthetic_input4model(data_array, sub_window_size, seq_length, synthetic_length)

    sequences = data_to_seq(array=data_array, seq_length=seq_length)

    # return sequences, synthetic_length
    return sequences


def make_prediction_from_dataLoader(ensemble_pre_trained_LSTM, models, dataLoader):
    array_temp = ensemble_pre_trained_LSTM.predictor_from_dataLoader(models, dataLoader)

    return array_temp


def make_prediction_from_sequences(ensemble_pre_trained_LSTM, models, sequences):
    array_temp = []

    for seq in tqdm(sequences,
                    total=len(sequences),
                    desc="Progress of <predictor_from_sequence>",
                    file=sys.stdout):
        t_features, features, t_target, target = seq
        t_str = UTCDateTime(t_target).strftime("%Y-%m-%dT%H:%M:%S")

        # predicted_pro is list
        predicted_pro, pro_mean, ci_range = ensemble_pre_trained_LSTM.predictor_from_sequence(features, t_features,
                                                                                              models)

        record = [float(f"{pro_mean:.3f}"), float(f"{ci_range:.3}")]
        record = [t_target, t_str, target] + predicted_pro + record  # merge the two lists
        array_temp.append(record)

    array_temp = np.array(array_temp)

    return array_temp


def plot_predicted_pro(benchmark_time, first_detection_str, st, array_temp, output_path, output_format, note):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import MultipleLocator

    plt.rcParams.update({'font.size': 7,
                         'axes.formatter.limits': (-4, 6),
                         'axes.formatter.use_mathtext': True})

    tr = st.copy()
    tr = stream_to_trace(st=tr)
    t1 = array_temp[0, 1]
    t2 = array_temp[-1, 1]
    tr.trim(UTCDateTime(t1), UTCDateTime(t2))

    tr = st.copy()
    tr = stream_to_trace(st=tr)
    t1 = UTCDateTime(year=tr.stats.starttime.year,
                     julday=tr.stats.starttime.julday,
                     hour=tr.stats.starttime.hour) + (tr.stats.starttime.minute + 1) * 60
    t2 = UTCDateTime(year=tr.stats.endtime.year,
                     julday=tr.stats.endtime.julday,
                     hour=tr.stats.endtime.hour) + (tr.stats.endtime.minute - 1) * 60
    tr.trim(UTCDateTime(t1), UTCDateTime(t2))

    # find the cloest time period
    id1 = np.argmin(np.abs(array_temp[:, 0].astype(float) - float(t1)))
    id2 = np.argmin(np.abs(array_temp[:, 0].astype(float) - float(t2))) + 1
    array_temp = array_temp[id1:id2, :]

    if array_temp.shape[0] > 720:  # 1 point is 1 minute, 720 is 12h
        x_interval = 6  # hour
    else:
        x_interval = 1  # hour

    fig = plt.figure(figsize=(5.5, 5))
    gs = gridspec.GridSpec(2, 1)

    ax = plt.subplot(gs[0])
    ax.set_title(output_format, loc="center", fontsize=7, fontweight="bold")
    sps = tr.stats.sampling_rate
    y = tr.data
    x = np.arange(len(y))
    ax.plot(x, y, color="black", zorder=2, label=f"{note}")
    ax.legend(fontsize=5)

    times = [benchmark_time, first_detection_str]
    labels = ["Benchmark Time\n(Max Amp. Time)", "First Detection"]
    colors_l = ["green", "red"]
    for t, l, c in zip(times, labels, colors_l):
        delta_t = UTCDateTime(t) - UTCDateTime(tr.stats.starttime)
        ax.axvline(x=delta_t * sps, color=c, label=l, zorder=1)

    ax.set_xlim(0, len(y))
    # ax.xaxis.set_major_locator(MultipleLocator(x_interval * sps))
    # ax.xaxis.set_minor_locator(MultipleLocator(x_interval * sps / 6))
    rewrite_x_ticks(ax, data_start=array_temp[0, 1], data_end=array_temp[-1, 1], data_sps=sps, x_interval=x_interval)
    ax.set_xlabel(f"Time from {array_temp[0, 1]} [1/{sps}]", fontweight='bold')
    ax.set_ylabel("Amplitude [m/s]", fontweight='bold')


    ax = plt.subplot(gs[1])
    sps = 1 / (UTCDateTime(array_temp[1, 1]) - UTCDateTime(array_temp[0, 1]))
    x = np.arange(len(array_temp[:, -2]))
    y = array_temp[:, -2].astype(float)
    ax.plot(x, y, color="black", alpha=0.5, zorder=3)
    ci_range = array_temp[:, -1].astype(float)

    y1 = y - ci_range
    y1 = np.clip(y1, a_min=0, a_max=1)
    y2 = y + ci_range
    y2 = np.clip(y2, a_min=0, a_max=1)

    ax.fill_between(x, y1, y2, color="black", alpha=0.5, zorder=2)

    times = [benchmark_time, first_detection_str]
    labels = ["Benchmark Time\n(Max Amp. Time)", "First Detection"]
    colors_l = ["green", "red"]
    for t, l, c in zip(times, labels, colors_l):
        delta_t = UTCDateTime(t) - UTCDateTime(tr.stats.starttime)
        ax.axvline(x=delta_t * sps, color=c, label=l, zorder=1)

    ax.set_xlim(0, len(y))
    # ax.xaxis.set_major_locator(MultipleLocator(x_interval * sps))
    # ax.xaxis.set_minor_locator(MultipleLocator(x_interval * sps / 6))
    rewrite_x_ticks(ax, data_start=array_temp[0, 1], data_end=array_temp[-1, 1], data_sps=sps, x_interval=x_interval)

    ax.set_xlabel(f"Time  from {array_temp[0, 1]} [minute]", fontweight='bold')
    ax.set_ylabel("Probability", fontweight='bold')
    ax.legend(fontsize=6)

    plt.tight_layout()
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(f"{output_path}/{output_format}.png", dpi=600)  # , transparent=True
    plt.close(fig=fig)
