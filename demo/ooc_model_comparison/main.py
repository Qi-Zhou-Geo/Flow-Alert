#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2025-10-14
# __author__ = Kshitij Kar, GFZ Helmholtz Centre for Geosciences
# __find me__ = kshitij.kar@gfz.de, kshitij787.ak@gmail.com, https://github.com/Kshitij301199
# Please do not distribute this code without the author's permission

import os
import sys
import shutil
import json
# Set CUDA environment variables
os.environ["CUDA_HOME"] = "/storage/vast-gfz-hpc-01/cluster/nvidia/cuda/11.6.2"
os.environ["PATH"] = os.path.join(os.environ["CUDA_HOME"], "bin") + ":" + os.environ.get("PATH", "")
os.environ["LD_LIBRARY_PATH"] = os.path.join(os.environ["CUDA_HOME"], "lib64") + ":" + os.environ.get("LD_LIBRARY_PATH", "")
import yaml

import torch
import numpy as np

from tqdm import tqdm

from obspy import Stream, UTCDateTime

# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path
current_dir = Path(__file__).resolve().parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys
sys.path.append(str(project_root))
# </editor-fold>

# import the custom functions
# from demo.Yanmen.remove_sensor_response import seismic_data_processing
from demo.ooc_model_comparison.seismic_signal_processing import load_seismic_signal

from functions.public.dataset_to_dataloader import *
from functions.public.prepare_feature4inference import Stream_to_feature
from functions.public.synthetic_input import synthetic_input4model
from functions.public.soft_normalize import soft_scaler

from functions.model.lstm_model import Ensemble_Trained_LSTM_Classifier
from functions.model.xlstm_model import Ensemble_Trained_xLSTM_Classifier

# INPUT PARAMETERS
sub_window_size = 30 # unit by second
window_overlap = 0 # 0 -> no overlap, 0.9 -> 90% overlap = 10% new data for each step

# num_extend = 4
# selected = None # this will be updated


# LSTM PARAMETERS
model_version_lstm = "v2model"
attention = True

# xLSTM PARAMETERS
model_version_xlstm = "v3model"

# DATA PARAMETERS
# continent = "Asian"
# region = "Bothekoshi"
# network = "XN"
# feature_type = "E"
# station_list = ["NEP04", "NEP05", "NEP06", "NEP07", "NEP08", "NEP10", "NEP13", "NEP16"]
# component = "HHZ"
# select_start_time = "2016-07-05T15:00:00"
# select_end_time = "2016-07-05T18:00:00"

# continent = "European"
# region = "Illgraben"
# network = "9S"
# feature_type = "E"
# station_list = ["ILL18", "ILL12", "ILL13"]
# component = "EHZ"
# select_start_time = "2020-06-08T13:55:00"
# select_end_time = "2020-06-08T18:00:00"

continent = "North_American"
region = "Mount_Hood"
network = "CC"
feature_type = "E"
station_list = ["PALM"]
component = "EHZ"
select_start_time = "2015-08-20T20:35:00"
select_end_time = "2015-08-20T21:34:36"

# continent = "North_American"
# region = "Montecito"
# network = "CI"
# feature_type = "E"
# station_list = ["QAD"]
# component = "HNZ"
# select_start_time = "2018-01-09T11:40:01"
# select_end_time = "2018-01-09T13:08:35"

# continent = "European"
# region = "Casamicciola"
# network = "IV"
# feature_type = "E"
# station_list = ["ICVJ"]
# component = "EHZ"
# select_start_time = "2022-11-26T04:00:41"
# select_end_time = "2022-11-26T04:04:01"

# continent = "North_American"
# region = "Mount_Baker"
# network = "UW"
# feature_type = "E"
# station_list = ["MBW"]
# component = "EHZ"
# select_start_time = "2013-05-31T09:52:00"
# select_end_time = "2013-05-31T10:15:11"

# continent = "North_American"
# region = "Museum_Fire"
# network = "1A"
# feature_type = "E"
# station_list = ["E19A", "COCB"]
# component = "CHZ"
# select_start_time = "2021-07-14T20:50:00"
# select_end_time = "2021-07-14T21:10:00"
# select_start_time = "2021-07-16T20:10:00"
# select_end_time = "2021-07-16T20:30:00"


year = UTCDateTime(select_start_time).year
start_hour = UTCDateTime(select_start_time).hour
end_hour = UTCDateTime(select_end_time).hour
event_julian_day = UTCDateTime(select_start_time).julday
start_time = str(UTCDateTime(year = year, julday = event_julian_day, hour=start_hour))
data_start_time = UTCDateTime(year = year, julday = event_julian_day, hour=max(start_hour-3, 0))
data_end_time = UTCDateTime(year = year, julday = event_julian_day, hour=min(end_hour+3, 23))
print(f"\nData Start : {data_start_time}, Data End : {data_end_time}\n")

# MODEL PARAMETERS GENERAL
batch_size = 128
seq_length = 32
num_repeat = 9

# LOAD MODELS
print(f"{'Loading Models':-^100}")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
## LSTM MODELS
ensemble_pre_trained_LSTM = Ensemble_Trained_LSTM_Classifier(model_version_lstm,
                                                             feature_type,
                                                             batch_size, seq_length,
                                                             device,
                                                             ML_name="LSTM", station="ILL02")

lstm_models = ensemble_pre_trained_LSTM.ensemble_models(num_repeat=num_repeat,
                                                   attention=attention,
                                                   print_model_summary=True)
## xLSTM MODELS
ensemble_pre_trained_xLSTM = Ensemble_Trained_xLSTM_Classifier(model_version_xlstm,
                                                               feature_type,
                                                               batch_size, seq_length,
                                                               device,
                                                               ML_name='xLSTM', station='ILL02')
xlstm_models = ensemble_pre_trained_xLSTM.ensemble_models(num_repeat=num_repeat,
                                                          print_model_summary=True)

print(f"{'Loaded Models':-^100}")
# LOAD SEISMIC DATA
print(f"{'Loading Seismic Data':-^100}")
st = Stream()
for station in tqdm(station_list, total=len(station_list), desc="Loading seismic data per station"):
    stt = load_seismic_signal(continent=continent,
                            region=region,
                            year=year,
                            network=network,
                            sta=station,
                            component=component,
                            julian_day=event_julian_day)
    stt.trim(starttime=data_start_time, endtime=data_end_time) # Apply to only the debris flow period, remove to apply to whole day
    st += stt
print(f"{'Loaded Seismic Data':-^100}")
print(f"{'Applying models':-^100}")

# if os.path.exists(f"{current_dir}/results/{continent}_{region}/{station}/"):
#     shutil.rmtree(f"{current_dir}/results/{continent}_{region}/{station}/")
# for station in station_list:
    # with open(f"{current_dir}/results/{station}_output.txt", "a") as output_file:
    #     output_file.write("Timestamp,LSTM_pro_mean,LSTM_pro_ci,xLSTM_pro_mean,xLSTM_pro_ci\n")
stream_to_feature = Stream_to_feature(sub_window_size, window_overlap)

config_path = f"{project_root}/config/config_inference.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

selected = np.array(config['feature_type_E'])

# PREPARE DATALOADER
print(f"{'Preparing Dataloader':-^100}")
for station in station_list:
    os.makedirs(f"{current_dir}/results/{continent}_{region}/{station}/", exist_ok=True)
    st_station = st.select(station=station)
    print(f"{f'Preparing Dataloader for {station}':-^100}")
    stream_to_feature = Stream_to_feature(sub_window_size, window_overlap)
    output_feature = stream_to_feature.prepare_feature(st_station, normalize_type="ref-itself")
    output_feature = np.column_stack((output_feature[:,1:2], output_feature[:,selected+2], np.zeros((output_feature.shape[0], 1)))) # Add fake label column
    sequences = data_to_seq(array=output_feature, seq_length=seq_length)
    dataset = seq_to_dataset(sequences=sequences, data_type="feature")
    dataloader = dataset_to_dataloader(dataset=dataset,
                                        batch_size=batch_size,
                                        training_or_testing="testing").dataLoader()
    print(f"{f'Dataloader prepared for {station}':-^100}")
    print(f"{f'Applying models for {station}':-^100}")
    print('\t\tLSTM Model')
    lstm_output = ensemble_pre_trained_LSTM.predictor_from_dataLoader(dataloader=dataloader, models=lstm_models)
    print('\t\txLSTM Model')
    xlstm_output = ensemble_pre_trained_xLSTM.predictor_from_dataLoader(dataloader=dataloader, models=xlstm_models)

    np.save(f"{current_dir}/results/{continent}_{region}/{station}/{year}_{event_julian_day}_lstm_output.npy", lstm_output)
    np.save(f"{current_dir}/results/{continent}_{region}/{station}/{year}_{event_julian_day}_xlstm_output.npy", xlstm_output)
    print(f"{f'Model results saved for {station}':-^100}")

print(f"\tResults saved in {current_dir}/results/")
print(f"{'End':-^100}")




