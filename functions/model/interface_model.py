#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2026-01-14
# __author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission

from typing import List

import os
import argparse

import yaml
import pickle

import joblib
import torch
import torch.nn as nn
# print("PyTorch version:", torch.__version__) = PyTorch version: 1.12.1
from torchinfo import summary
# print("Torchinfo version:", torchinfo.__version__) = Torchinfo version: 1.8.0


import numpy as np
import pandas as pd

from tqdm import tqdm

from obspy import UTCDateTime


# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path
current_dir = Path(__file__).resolve().parent

# using ".parent" on "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys
sys.path.append(str(project_root))
# </editor-fold>

# import the custom functions
from functions.model.lstm_model import LSTM_Attention
from functions.data_process.prepare_feature4inference import Stream_to_feature
from functions.seismic.st2tr import stream_to_trace
from functions.data_process.dataset_to_dataloader import data_to_seq
from functions.statistical_test.confidence_level_test import student_t_testing
from functions.data_process.cross_catchments_inference import find_max_amp_time
from functions.warning_strategy.calculate_inference_matrix import inference_matrix
from functions.data_process.load_data import clip_df_columns

def load_pretrained_models(model_type: str = "LSTM",
                           model_version: str = "v1dot3",
                           inference_model_config: dict = None,
                           device: str = "cpu"):
    '''
    Load pre-trained models in List.

    Args:
        model_version: str, requested model version

    Returns:
        model_list: List[nn.Module] or List[pkl model]

    '''

    if model_version in ["v1dot3", "v1.3", "v1dot3model"]:
        # it should like this: "Model=[ML_name]_STA=[station]_Feature=[feature_type]_repeat=[num_repeat]"
        archived_model_format = inference_model_config["archived_model_format"]
        archived_model_format = archived_model_format.replace("[ML_name]", model_type)
        for i in ["station", "feature_type"]:
            archived_model_format = archived_model_format.replace(f"[{i}]", inference_model_config[i])
        # except the "num_repeat", all params are repalced now.
    else:
        raise ValueError(f"model_version={model_version} not supported")

    # the path of the saved model
    ref_model_dir = f"{project_root}/trained_model/{model_version}"
    model_list = []
    for repeat in range(1, inference_model_config[model_type]["num_repeat"] + 1):

        # repalce the repeat
        archived_model_format = archived_model_format.replace("[num_repeat]", str(repeat))
        full_path = f"{ref_model_dir}/{archived_model_format}.{inference_model_config[model_type]['extension']}"

        # load the model from local path
        if model_type in ["RF", "Random_Forest"]:
            model = joblib.load(f"{full_path}")
        elif model_type in ["XGB", "XGBoost"]:
            from xgboost import XGBClassifier
            model = XGBClassifier()
            model.load_model(full_path)
        elif model_type in ["LSTM"]: # for LSTM model
            model = LSTM_Attention(feature_size=inference_model_config["feature_size"], device=device)
            load_checkpoint = torch.load(f"{full_path}", map_location=torch.device('cpu'))
            model.load_state_dict(load_checkpoint)
            model.to(device)
            model.eval()  # set as "evaluate" mode

            # print the model structure
            if repeat == 1:
                batch_size = 1
                seq_length = inference_model_config[model_type]["seq_length"]
                feature_size = inference_model_config["feature_size"]

                s = summary(model=model,
                            input_size=[(batch_size, seq_length, feature_size), (batch_size, seq_length)],
                            col_names=("input_size", "output_size", "num_params", "params_percent", "trainable"),
                            device=device)
                print(f"Load Pre-trained model from:\n <{full_path}>. \n"
                      f"Model summary:\n {s}.\n")
        else:
            print(f"model_version={model_version} not supported")

        model_list.append(model)

    return model_list

def plot_predicted_pro(benchmark_time,
                       first_detection_str,
                       st,
                       array_temp,
                       output_path,
                       output_format,
                       note):

    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import MultipleLocator
    from functions.visualize.visualize_seismic import rewrite_x_ticks

    plt.rcParams.update({'font.size': 7,
                         'axes.formatter.limits': (-4, 6),
                         'axes.formatter.use_mathtext': True})

    tr = st.copy()
    tr = stream_to_trace(st=tr)
    t1 = array_temp[0, 1]
    t2 = array_temp[-1, 1]
    # select the st length based on the model output (array_temp),
    # st could longer than model predict, bacasue the sequence prepration
    tr.trim(UTCDateTime(t1), UTCDateTime(t2))

    # re-selsct the st from minute level
    t1 = UTCDateTime(year=tr.stats.starttime.year,
                     julday=tr.stats.starttime.julday,
                     hour=tr.stats.starttime.hour) + (tr.stats.starttime.minute + 1) * 60
    t2 = UTCDateTime(year=tr.stats.endtime.year,
                     julday=tr.stats.endtime.julday,
                     hour=tr.stats.endtime.hour) - (tr.stats.endtime.minute - 1) * 60
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
    sps = round(tr.stats.sampling_rate)
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
    rewrite_x_ticks(ax, data_start=array_temp[0, 1], data_end=array_temp[-1, 1], data_sps=sps, x_interval=x_interval)

    ax.set_xlabel(f"Time from {array_temp[0, 1]} [minute]", fontweight='bold')
    ax.set_ylabel("Probability", fontweight='bold')
    ax.legend(fontsize=6)

    plt.tight_layout()
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(f"{output_path}/{output_format}.png", dpi=600)  # , transparent=True
    plt.show()
    plt.close(fig=fig)


class FlowAlert:

    def __init__(self, model_type, model_version,
                 st, output_path,
                 sub_window_size=60, window_overlap=0):

        # model params
        self.model_type = model_type
        self.model_version = model_version


        # seismic data and desired window length
        self.st = stream_to_trace(st) # convert to trace
        st_duration = UTCDateTime(self.st.stats.endtime) - UTCDateTime(self.st.stats.starttime)
        assert st_duration > 3600 * 3, f"Warning!\n Please input longer (> 3h) seismic stream, now it is {st_duration} seconds long."

        self.sub_window_size = sub_window_size # unit by second
        self.window_overlap = window_overlap # # unit by ratio, 0-> none overlap, 1-> fully overlap


        # FlowAlert data
        self.inference_model_config = None # Dict
        self.model_list = None # List of trained model
        self.model_input = None # numpy array, [float timestamps, features, labels]
        self.model_output = None # numpy array, [float timestamps, str timestamps, pro1-N, pro_mean, pro_CI]

        # hardware
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # I/O path
        self.project_root = project_root
        self.output_path = output_path
        self.output_format = (f"{self.st.stats.network}-{self.st.stats.station}-{self.st.stats.channel}-"
                              f"{self.model_type}-{self.model_version}-{self.sub_window_size}-{self.window_overlap}")

        # Auto-initialize
        # self.model_config()
        # self.model()
        # self.feature()
        # self.prediction()

    def model_config(self):
        if self.model_version in ["v1dot3", "v1.3", "v1dot3model"]:
            config_path = f"{self.project_root}/config/config_v1dot3model.yaml"
            with open(config_path, "r") as f:
                inference_model_config = yaml.safe_load(f)

            self.inference_model_config = inference_model_config

        return inference_model_config

    def load_model(self):

        model_list = load_pretrained_models(model_type=self.model_type,
                                            model_version=self.model_version,
                                            inference_model_config=self.inference_model_config,
                                            device=self.device)
        # List[models]
        self.model_list = model_list

        return model_list

    def prepare_feature(self, sub_window_size=None, window_overlap=None):

        # with these params, you can use the different window length for FlowAlert
        # the default is 60 s without overlap
        if sub_window_size is None:
            sub_window_size = self.sub_window_size
        else:
            sub_window_size = sub_window_size

        if window_overlap is None:
            window_overlap = self.window_overlap
        else:
            window_overlap = window_overlap

        feature_type = self.inference_model_config["feature_type"]
        st2f = Stream_to_feature(sub_window_size, window_overlap, feature_type)
        output_feature = st2f.prepare_feature_mpi(st=self.st, num_cpus=6) # make sure you have 6 CPUs

        feature_arr = output_feature[:, :-3].astype(float)
        t_str = output_feature[:, -3]
        t_float = output_feature[:, -2].astype(float).reshape(-1, 1)
        pretend_label = output_feature[:, -1].astype(float).reshape(-1, 1)

        # 2D ([float time stamps, features, labels])
        feature_arr = np.concatenate((t_float, feature_arr, pretend_label), axis=1)

        # 2D ([float time stamps, features 1->N, labels])
        self.model_input = feature_arr

        return feature_arr

    def make_prediction(self, tested_model=None):

        # with this param, you can use the same FlowAlert class to test multiple models
        if tested_model is None:
            tested_model = self.model_type
        else:
            tested_model = tested_model

        # pass the data to model
        if tested_model in ["RF", "Random_Forest", "XGB", "XGBoost"]:
            data = self.model_input
            model_output = self.by_tree_model(data=data)
        else:
            seq_length = self.inference_model_config[self.model_type]["seq_length"]
            # convert numpy as data sequence for LSTM model
            data = data_to_seq(array=self.model_input, seq_length=seq_length)
            model_output = self.by_dl_model(data=data)

        return model_output

    def by_tree_model(self, data):

        t_float = data[:, 0].reshape(-1, 1)
        t_str = np.array([UTCDateTime(i).isoformat() for i in t_float]).reshape(-1, 1)
        feature_arr = data[:, 1:-1]
        target = data[:, -1]  # do not need now (2026-01-19)

        temp_pro = []
        for model in self.model_list:
            predicted_pro = model.predict_proba(feature_arr)[:, 1] # only select the DF pro
            pre_y_pro = np.round(predicted_pro, decimals=3).astype(float)
            # model predicted train_data label
            pre_y_label = model.predict(feature_arr).astype(float) # do not need now (2026-01-19)
            temp_pro.append(pre_y_pro)

        temp_pro = np.column_stack(temp_pro) # stack as column, not like LSTM model

        # do the stastic test
        input_data = temp_pro
        output_mean, output_ci_range = student_t_testing(input_data=input_data,
                                                         row_or_column="row", # for each time step
                                                         confidence_interval=0.95)
        output_mean, output_ci_range = output_mean.reshape(-1, 1), output_ci_range.reshape(-1, 1)

        # prepare the output
        model_output = np.concatenate((t_float, t_str, temp_pro, output_mean, output_ci_range), axis=1)
        self.model_output = model_output

        return model_output

    def by_dl_model(self, data):

        batch_size = 1 # do not use large batch
        seq_length = self.inference_model_config[self.model_type]["seq_length"]
        feature_size = self.inference_model_config["feature_size"]

        temp_pro = [] # store all time stamps

        # 1st loop all the seq in time doamin
        for seq in tqdm(data,
                        total=len(data),
                        desc="Progress of <prediction_from_sequence>",
                        file=sys.stdout):

            t_features, feature_arr, t_target, target = seq

            # reshape the feature_arr as tensor with shape [batch_size, seq_length, feature_size]
            t_features = (torch.from_numpy(t_features)).to(self.device)
            feature_arr = feature_arr.reshape(batch_size, seq_length, feature_size).astype(np.float32)
            feature_arr = (torch.from_numpy(feature_arr)).to(self.device)

            # 2nd loop all the same seq in model doamin
            predicted_pro = []
            for model in self.model_list:
                # make sure does not change the model parameters
                model = model.to(self.device)
                model.eval()
                with torch.no_grad():
                    # return the model output logits, shape (batch_size, 2)
                    raw_logits = model(feature_arr, t_features)
                    DF_pro = torch.softmax(raw_logits, dim=1)[:, 1] # # only select the DF pro
                    DF_pro = DF_pro.cpu().detach().numpy()
                    predicted_pro.append(DF_pro)

            # prepare the current output
            t_float = t_target
            t_str = UTCDateTime(t_target).isoformat()
            predicted_pro = [np.round(arr[0].item(), 3) for arr in predicted_pro]
            predicted_pro = [t_float, t_str] + predicted_pro

            # each row represent one time step, not like tree model
            temp_pro.append(predicted_pro)

        temp_pro = np.row_stack(temp_pro)

        # do the stastic test
        input_data = temp_pro[:, 2:].astype(float) # the shape[1] should == len(model_input)
        output_mean, output_ci_range = student_t_testing(input_data=input_data ,
                                                         row_or_column="row", # for each time step
                                                         confidence_interval=0.95)
        output_mean, output_ci_range = output_mean.reshape(-1, 1), output_ci_range.reshape(-1, 1)

        # prepare the output
        model_output = np.concatenate((temp_pro, output_mean, output_ci_range), axis=1)
        self.model_output = model_output

        return model_output

    def plot(self, event_start=None, event_end=None, benchmark_time=None):

        # define the time stamps (isoformat)
        if event_start is None:
            event_start = self.st.stats.starttime.isoformat()
        else:
            event_start = UTCDateTime(event_start).isoformat()

        if event_end is None:
            # use the data end as envent end time
            event_end = self.st.stats.endtime.isoformat()
        else:
            event_end = UTCDateTime(event_end).isoformat()

        if benchmark_time is None:
            # use the max amplitude during the amplitude as max time
            benchmark_time = find_max_amp_time(self.st, event_start, event_end)
        else:
            benchmark_time = UTCDateTime(benchmark_time).isoformat()

        # the array_temp should shape by [float timestamps, str timestamps, pro1-N, pro_mean, pro_CI]
        array_temp = self.model_output

        temp = inference_matrix(array_temp, benchmark_time, event_start, event_end,
                                pro_epsilon=0.5, buffer1=3, buffer2=3)
        detection_type, first_detection, first_detection_str, increased_warning_time, false_detection, false_detection_ratio = temp
        note = (f"params: {self.output_format},\n"
                f"detection type: {detection_type},\n"
                f"first detection time: {first_detection_str},\n"
                f"benchmark time: {benchmark_time},\n"
                f"increased warning time: {increased_warning_time} [seconds],\n"
                f"false detection ratio: {false_detection_ratio:.3f}")

        plot_predicted_pro(benchmark_time,
                           first_detection_str,
                           self.st,
                           array_temp,
                           self.output_path,
                           self.output_format,
                           note)

        time_now = UTCDateTime.now().isoformat()
        print(f"{time_now}\n{note}")