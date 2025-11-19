#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2025-05-30
# __author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission

import yaml
import os

import numpy as np

from tqdm import tqdm

from datetime import datetime, timezone

from obspy import Stream, UTCDateTime, read

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path
current_dir = Path(__file__).resolve().parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys
sys.path.append(str(project_root))
# </editor-fold>

# import the custom functions
from functions.seismic.seismic_data_processing import load_seismic_signal
from functions.public.dataset_to_dataloader import *
from functions.public.prepare_feature4inference import Stream_to_feature
from functions.model.lstm_model import Ensemble_Trained_LSTM_Classifier
from functions.public.synthetic_input import synthetic_input4model
from functions.warning_strategy.strategy import warning_controller


# <editor-fold desc="prepare the input parameters">
sub_window_size = 60 # unit by second
window_overlap = 0 # 0 -> no overlap, 0.9 -> 90% overlap = 10% new data for each step

seq_length = 32
num_extend = 4
selected = None # this will be updated

select_start_time =  "2025-07-18T01:00:00"
select_end_time =  "2025-07-18T23:50:00"
station_list = ["3018849"]

attention_window_size = 10
warning_threshold = 0.5

model_type = "LSTM"
feature_type = "E"
trained_model_name = f"{model_type}_{feature_type}"
model_version, num_repeat, attention = "v2model", 7, True
batch_size = 128
seq_length = 32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# </editor-fold>


# <editor-fold desc="load seismic data">
st = read(f"{project_root}/demo/Jiangjia/453018849.0037.2025.07.18.00.00.00.000.Z.sac")
# st = read(f"/Users/qizhou/Desktop/DC.BL9.BHZ.2023.214.mseed")
st.merge(method=1, fill_value='latest', interpolation_samples=0)
st._cleanup()
st.detrend('linear')
st.detrend('demean')
st.plot()

paz = {  # 2025 jiangjia smart solo
    'zeros': [(0 + 0j),
              (0 + 0j)],

    'poles': [(-22.211059 + 22.217768j),
              (-22.211059 - 22.217768j)],
    'gain': 1000,
    'sensitivity': 76.7  # V / (m/s)
}
pre_filt=(0.5, 1.0, 100, 110.0)

st.simulate(
    paz_remove=paz ,
    paz_simulate=None,
    remove_sensitivity=True,
    pre_filt=pre_filt
)

st.detrend('linear')
st.detrend('demean')
st.plot()
# </editor-fold>


# <editor-fold desc="prepare the Stream_to_feature">
stream_to_feature = Stream_to_feature(sub_window_size, window_overlap)
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

selected = config['feature_type_E']
# </editor-fold>

synthetic_feature = synthetic_input4model(sub_window_size, window_overlap, trained_model_name, seq_length*num_extend)
pro_arr = np.full((attention_window_size, len(station_list)), fill_value=0, dtype=float)
pro_ci_arr = pro_arr.copy()
delta_t = sub_window_size * (1 - window_overlap)

num_step = UTCDateTime(select_end_time) - UTCDateTime(select_start_time)
num_step = num_step / delta_t

pro_temp = []
pro_ci_temp = []

for t in tqdm(np.arange(1, num_step)):

    tr = st.copy()
    t1 = UTCDateTime(select_start_time) + t * delta_t
    t2 = UTCDateTime(select_start_time) + (t + 1) * delta_t
    tr.trim(t1, t2)

    # calculate the features
    output_feature = stream_to_feature.one_step_feature(tr=tr, normalize_type=None)
    feature_arr = output_feature[:, 2:][:, selected]  # only selected the features that pretrained model needed
    output_feature = np.concatenate((output_feature[:, :2], feature_arr), axis=1)  # merge in column dimenssion

    # update the synthetic_feature
    synthetic_feature = np.vstack((synthetic_feature, output_feature.reshape(1, -1)))  # add the new time feature
    synthetic_feature = synthetic_feature[-seq_length * num_extend:, :]  # remove the oldest time feature

    # prepare normalized feature
    t_features = synthetic_feature[:, 1].copy().astype(float)
    features = synthetic_feature[:, 2:].copy().astype(float)

    # data_arr = np.load(f"{current_dir}/min_max/{station}_values.npz", allow_pickle=True)
    # min_arr = data_arr["min_arr"]
    # max_arr = data_arr["max_arr"]
    #
    # features = (features - min_arr) / (max_arr - min_arr)  # normalize
    features = features[-seq_length:, :]

    # make the prediction by seismic feature
    predicted_pro, pro_mean, ci_range = ensemble_pre_trained_LSTM.predictor_from_sequence(features, t_features, models)

    # update the matrix
    pro_temp.append(np.round(pro_mean, 4))
    pro_ci_temp.append(np.round(ci_range, 4))



# plot
plt.rcParams.update( {'font.size':7,
                      'axes.formatter.limits': (-2, 3),
                      'axes.formatter.use_mathtext': True} )

fig = plt.figure(figsize=(5.5, 3))
gs = gridspec.GridSpec(1, 1)
ax = plt.subplot(gs[0])

pre_y_pro = np.array(pro_temp)
ci_range = np.array(pro_ci_temp)
ci_lower = pre_y_pro - ci_range
ci_lower[ci_lower<0] = 0
ci_upper = pre_y_pro + ci_range

x = np.arange(len(pre_y_pro))

plt.plot(x, pro_temp, color="black",  label="Mean DF Pro.", zorder=1)
plt.fill_between(x, ci_lower, ci_upper, color="black", label="95% CI", alpha=0.5, zorder=2)
plt.legend()


t1 = datetime.strptime(select_start_time, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
t2 = datetime.strptime(select_end_time, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()

duration = int((t2-t1) / 3600)
sps_data = 1/sub_window_size
x_interval = 4 # hour
xLocation = np.arange(0, sps_data * 3600 * (duration + x_interval), sps_data * 3600 * x_interval)
xTicks = []
for idx, i in enumerate(xLocation):
    if idx == 0:
        t_temp = datetime.fromtimestamp(t1 + i * 1 / sps_data, tz=timezone.utc).strftime('%Y-%m-%d' + '\n' + '%H:%M:%S')
    else:
        t_temp = datetime.fromtimestamp(t1 + i * 1 / sps_data, tz=timezone.utc).strftime('%H:%M:%S')
    xTicks.append(t_temp)

ax.set_xticks(xLocation, xTicks)
ax.set_xlabel("Time [UTC+8]", weight='bold')
ax.set_ylabel("Probability", weight='bold')

plt.tight_layout()
plt.savefig(f"{current_dir}/Jiangjia_2025.png", dpi=600)
plt.show()
