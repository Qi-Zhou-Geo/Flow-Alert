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

from datetime import datetime, timezone

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
from functions.public.df_to_dataloader import prepare_dataloader, prepare_sequences
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
model_version, num_repeat, attention = "v2model", 9, True
feature_type = "E"
batch_size = 128
seq_length = 32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# </editor-fold>


# <editor-fold desc="load the pre-trained model">
ensemble_pre_trained_LSTM = Ensemble_Trained_LSTM_Classifier(model_version,
                                                             feature_type,
                                                             batch_size, seq_length,
                                                             device,
                                                             ML_name="LSTM", station="ILL02")

models = ensemble_pre_trained_LSTM.ensemble_models(num_repeat=num_repeat,
                                                   attention=attention,
                                                   print_model_summary=True)
# </editor-fold>


# <editor-fold desc="select the feature ID for the pre-trained model">
config_path = f"{project_root}/config/config_inference.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

selected = config[f'feature_type_{feature_type}']
# </editor-fold>


# <editor-fold desc="load the seismic features">
# </editor-fold>

params_list = [
"Luding-WD-2023-STA02-BHZ-E-testing-False",
"Luding-LD-2023-STA01-DPZ-E-testing-False",
"Luding-AM-2024-R9BF5-EHZ-E-testing-False",
]

normalize = True
for params in params_list:
    params = [params]
    _, test_sequences = prepare_sequences(params, normalize, seq_length)

    # <editor-fold desc="pre-trained model inference">
    # </editor-fold>
    to_be_saved = []
    for seq in tqdm(test_sequences, desc="Progress of <predictor_from_sequence>", file=sys.stdout):
        t_features, features, t_target, target = seq
        # t_str = datetime.fromtimestamp(t_target, tz=pytz.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')

        # predicted_pro is a list
        predicted_pro, pro_mean, ci_range = \
            ensemble_pre_trained_LSTM.predictor_from_sequence(features, t_features, models)
        record = [float(f"{pro_mean:.3f}"), float(f"{ci_range:.3}")]
        record = [t_target, target] + predicted_pro + record  # merge the two lists
        to_be_saved.append(record)

    to_be_saved = np.array(to_be_saved)
    date_str = [datetime.fromtimestamp(i, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") for i in to_be_saved[:, 0]]
    date_str = np.array(date_str).reshape(-1, 1)
    to_be_saved = np.hstack((date_str, to_be_saved))

    pro_header= [f"pro{i}" for i in range(1, num_repeat+1)]
    header = f"time_stamps_str, time_stamps,target,{','.join(pro_header)},pro_mean,pro_95ci"
    np.savetxt(f"{current_dir}/{params[0]}-predicted.txt",
               to_be_saved, delimiter=",", fmt="%s", header=header, comments='')
