#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2025-05-30
# __author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission

import sys
import yaml

import torch
import numpy as np

from tqdm import tqdm

from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec


from obspy.clients.fdsn import Client
from obspy import read, Stream, read_inventory, signal, UTCDateTime

# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path
current_dir = Path(__file__).resolve().parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys
sys.path.append(str(project_root))
# </editor-fold>

# import the custom functions
from demo.Yanmen.remove_sensor_response import seismic_data_processing

from functions.public.load_data import select_features
from functions.public.dataset_to_dataloader import *
from functions.public.prepare_feature4inference import Stream_to_feature
from functions.public.prepare_SNR4inference import Stream_to_matrix
from functions.model.lstm_model import Ensemble_Trained_LSTM_Classifier
from functions.seismic.generate_seismic_trace import create_trace
from functions.seismic.plot_obspy_st import time_series_plot
from functions.warning_strategy.calculate_inference_matrix import inference_matrix
from functions.public.synthetic_input import synthetic_input4model
from functions.seismic.visualize_seismic import psd_plot, waveform_plot, pro_plot, convert_st2tr
from functions.warning_strategy.strategy import warning_controller


# <editor-fold desc="prepare the input parameters">
sub_window_size = 30 # unit by second
window_overlap = 0 # 0 -> no overlap, 0.9 -> 90% overlap = 10% new data for each step

<<<<<<< HEAD
normalize_type = "ref-itself"
trained_model_name = "LSTM_E"
=======
>>>>>>> test
seq_length = 32
num_extend = 4
selected = None # this will be updated

select_start_time =  "2025-07-02T02:00:00"
select_end_time =  "2025-07-04T22:00:00"

station_list = ["STA01", "STA02", "STA03", "STA04", "STA05",
                "STA07", "STA08"]


attention_window_size = 10
warning_threshold = 0.5

<<<<<<< HEAD
trained_model_name, model_version, num_repeate, attention = "LSTM_E", "v2-model", 7, True
=======
model_type = "LSTM"
feature_type = "E"
trained_model_name = f"{model_type}_{feature_type}"
model_version, num_repeat, attention = "v2model", 3, True
batch_size = 128
seq_length = 32
>>>>>>> test
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# </editor-fold>


# <editor-fold desc="load seismic data">
sac_path = "/storage/vast-gfz-hpc-01/project/seismic_data_qi/seismic/Asian/Yanmen/2025" # '/Volumes/GFZsec47/Yanmen' #
st = Stream()

data_path = f'{sac_path}/7.2'
for s in np.arange(1, 9):
    data_name = f'Point {s}.HHZ.sac'
    tr = seismic_data_processing(data_path, data_name)
    st += tr


data_path = f'{sac_path}/7.3'
for s in np.arange(1, 9):
    data_name = f'Point {s}.HHZ.sac'
    tr = seismic_data_processing(data_path, data_name)
    st += tr

data_path = f'{sac_path}/7.4'
for s in np.arange(1, 9):
    data_name = f'Point {s}.HHZ.sac'
    tr = seismic_data_processing(data_path, data_name)
    st += tr


st.plot()
st.merge(method=1, fill_value='latest', interpolation_samples=0)
st._cleanup()
# </editor-fold>


# <editor-fold desc="prepare the Stream_to_feature">
stream_to_feature = Stream_to_feature(sub_window_size, window_overlap)
# </editor-fold>


# <editor-fold desc="load the pre-trained model">
<<<<<<< HEAD
ensemble_pre_trained_LSTM = Ensemble_Trained_LSTM_Classifier(trained_model_name, model_version, device)
models = ensemble_pre_trained_LSTM.ensemble_models(num_repeate=num_repeate,
                                                   attention=attention,
                                                   print_model_summary=True)

=======
ensemble_pre_trained_LSTM = Ensemble_Trained_LSTM_Classifier(model_version,
                                                             feature_type,
                                                             batch_size, seq_length,
                                                             device,
                                                             ML_name="LSTM", station="ILL02")

models = ensemble_pre_trained_LSTM.ensemble_models(num_repeat=num_repeat,
                                                   attention=attention,
                                                   print_model_summary=True)
>>>>>>> test
# </editor-fold>


# <editor-fold desc="select the feature ID for the pre-trained model">
<<<<<<< HEAD
if trained_model_name == "LSTM_E":
    config_path = f"{project_root}/config/config_inference.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    selected = config['feature_type_E']
=======
config_path = f"{project_root}/config/config_inference.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

selected = config[f'feature_type_{feature_type}']
>>>>>>> test
# </editor-fold>

synthetic_feature = synthetic_input4model(sub_window_size, window_overlap, trained_model_name, seq_length*num_extend)
pro_arr = np.full((attention_window_size, len(station_list)), fill_value=0, dtype=float)
pro_ci_arr = pro_arr.copy()
delta_t = sub_window_size * (1 - window_overlap)

num_step = UTCDateTime(select_end_time) - UTCDateTime(select_start_time)
num_step = num_step / delta_t

for t in tqdm(np.arange(1, num_step)):

    stt = st.copy()
    t1 = UTCDateTime(select_start_time) + t * delta_t
    t2 = UTCDateTime(select_start_time) + (t + 1) * delta_t
    stt.trim(t1, t2)

    pro_temp = []
    pro_ci_temp = []
    for s, station in enumerate(station_list):

        tr = stt.copy()
        tr = tr.select(station=station)

        # <editor-fold desc="convert seismic stream to seismic feature">
        output_feature = stream_to_feature.one_step_feature(tr=tr, normalize_type=None)

        feature_arr = output_feature[:, 2:][:, selected] # only selected the features that pretrained model needed
        output_feature = np.concatenate((output_feature[:, :2], feature_arr), axis=1) # merge in column dimenssion

        #with open(f"{current_dir}/{station}.txt", mode="a") as f:
            #temp = feature_arr.tolist()
            #temp = f'{stt[0].stats.starttime.strftime("%Y-%m-%dT%H:%M:%S")}, ' \
                   #f'{", ".join(map(str, temp))}\n'
            #f.write(temp)

        # update the synthetic_feature
        synthetic_feature = np.vstack((synthetic_feature, output_feature.reshape(1, -1))) # add the new time feature
        synthetic_feature = synthetic_feature[-seq_length * num_extend:, :]  # remove the oldest time feature

        # prepare normalized feature
        t_features = synthetic_feature[:, 1].copy().astype(float)
        features = synthetic_feature[:, 2:].copy().astype(float)

        #min_vals = features.min(axis=0)
        #max_vals = features.max(axis=0)
        #range_vals = np.where((max_vals - min_vals) == 0, 1, max_vals - min_vals)
        data_arr = np.load(f"{current_dir}/min_max/{station}_values.npz", allow_pickle=True)
        min_arr = data_arr["min_arr"]
        max_arr = data_arr["max_arr"]

        features = (features - min_arr) / (max_arr - min_arr) # normalize
        features = features[-seq_length:, :]

        # </editor-fold>

        # <editor-fold desc="make the prediction by seismic feature">
        predicted_pro, pro_mean, ci_range = ensemble_pre_trained_LSTM.predictor_from_sequence(features, t_features, models)
        # </editor-fold>

        # update the matrix
        pro_temp.append(np.round(pro_mean, 4))
        pro_ci_temp.append(np.round(ci_range, 4))

    pro_temp = np.array(pro_temp).reshape(1, len(station_list))
    pro_arr = np.vstack((pro_arr, pro_temp))
    pro_arr = pro_arr[-attention_window_size:, :]

    pro_ci_temp = np.array(pro_ci_temp).reshape(1, len(station_list))
    pro_ci_arr = np.vstack((pro_arr, pro_ci_temp))
    pro_ci_arr = pro_ci_arr[-attention_window_size:, :]

    # for upstream STA01 to STA05
    status_up = warning_controller(pro_arr[:, :-2], warning_threshold, pro_filter=0)
    # for downstream STA07 and STA08
    status_down = warning_controller(pro_arr[:, -2:], warning_threshold, pro_filter=0)

    if status_up == "warning":
        print(f'Warning!!!'
              f'\nTime: UTC+8 {stt[0].stats.starttime.strftime("%Y-%m-%dT%H:%M:%S")}, '
              f'\nLocation: STA01 to STA05,'
              f'Warning Threshold: {warning_threshold}')

    if status_down == "warning":
        print(f'Warning!!!'
              f'\nTime: UTC+8 {stt[0].stats.starttime.strftime("%Y-%m-%dT%H:%M:%S")}, '
              f'\nLocation: STA07 and STA08,'
              f'Warning Threshold: {warning_threshold}')


    with open(f"{current_dir}/test.txt", mode="a") as f:
        pro = pro_arr[-1, :].tolist()
        temp = f'{stt[0].stats.starttime.strftime("%Y-%m-%dT%H:%M:%S")}, ' \
               f'{", ".join(map(str, pro))}, {status_up}, {status_down}\n'
        f.write(temp)
<<<<<<< HEAD

=======
>>>>>>> test
