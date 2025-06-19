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
project_root = current_dir.parent
import sys
sys.path.append(str(project_root))
# </editor-fold>

# import the custom functions
from functions.public.load_data import select_features
from functions.public.dataset_to_dataloader import *
from functions.public.prepare_feature4inference import Stream_to_feature
from functions.public.prepare_SNR4inference import Stream_to_matrix
from functions.model.lstm_model import Ensemble_Trained_LSTM_Classifier
from functions.seismic_data_processing_obspy.generate_seismic_trace import create_trace
from functions.seismic_data_processing_obspy.plot_obspy_st import time_series_plot
from functions.issue_network_warning.calculate_inference_matrix import inference_matrix


# <editor-fold desc="load the seismic data">
def fetch_data(client_name, start_time, end_time, time_buffer,
               network, station, location, channel,
               f_min, f_max):

    client = Client(client_name)
    start_time = UTCDateTime(start_time) - 3600 * time_buffer
    end_time = UTCDateTime(end_time) + 3600 * time_buffer

    inv = client.get_stations(starttime=start_time, endtime=end_time,
                              network=network, station=station,
                              location=location, channel=channel,
                              level="response", format="xml")

    st = client.get_waveforms(network=network, station=station,
                              location=location, channel=channel,
                              starttime=start_time, endtime=end_time)

    st.merge(method=1, fill_value='latest', interpolation_samples=0)
    st._cleanup()
    st.detrend('linear')
    st.detrend('demean')
    st.remove_response(inventory=inv)

    st.filter("bandpass", freqmin=f_min, freqmax=f_max)
    st.detrend('linear')
    st.detrend('demean')

    st.trim(starttime=start_time, endtime=end_time, nearest_sample=False)

    return st

client_name = "ETH"
start_time, end_time, time_buffer = "2025-05-28T12:00:00", "2025-05-28T15:00:00", 0
#network, station, location, channel ="CH",  "LAUCH", "", "HHZ"
#network, station, location, channel ="CH",  "MUGIO", "", "HHZ"
#network, station, location, channel ="CH",  "BERNI", "", "HHZ"
network, station, location, channel ="CH",  "FIESA", "", "HHZ"

f_min, f_max = 1, 5

st = fetch_data(client_name, start_time, end_time, time_buffer,
               network, station, location, channel,
               f_min, f_max)
st.plot()
# </editor-fold>


# <editor-fold desc="convert seismic stream to seismic feature">
# config the pre-trained parameters
sub_window_size, window_overlap = 20, 0.5
normalize_type = "ref-itself"
trained_model_name = "LSTM_E"
seq_length = 32

# seismic matrix
stream_to_matrix = Stream_to_matrix(sub_window_size, window_overlap)
output_matrix = stream_to_matrix.prepare_matrix(st=st, print_reminder=True)


# seismic features
stream_to_feature = Stream_to_feature(sub_window_size, window_overlap)
output_feature = stream_to_feature.prepare_feature(st=st, normalize_type=normalize_type)
t_str, t_float, feature_arr = output_feature[:, 0], output_feature[:, 1], output_feature[:, 2:]


if trained_model_name == "LSTM_E":
    config_path = f"{project_root}/config/config_inference.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    selected = config['feature_type_E']
    feature_arr = feature_arr[:, selected]

t_float = t_float.astype(float).reshape(-1, 1)
data_array = np.hstack((t_float, feature_arr))  # 2D ([time stamps, features])
pretend_label = np.full(data_array.shape[0], 0).reshape(-1, 1)  # for the event we do not have benchmark time
data_array = np.hstack((data_array, pretend_label))  # 2D ([time stamps, features, labels])
sequences = data_to_seq(array=data_array, seq_length=seq_length)
# </editor-fold>


# <editor-fold desc="load the pre-trained model">
trained_model_name, model_version, num_repeate, attention = "LSTM_E", "v2-model", 7, True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ensemble_pre_trained_LSTM = Ensemble_Trained_LSTM_Classifier(trained_model_name, model_version, device)
models = ensemble_pre_trained_LSTM.ensemble_models(num_repeate=num_repeate, attention=attention, print_model_summary=True)
# </editor-fold>


# <editor-fold desc="make the prediction by seismic feature">
to_be_saved = []
for seq in tqdm(sequences, desc="Progress of <predictor_from_sequence>", file=sys.stdout):
    t_features, features, t_target, target = seq
    # t_str = datetime.fromtimestamp(t_target, tz=pytz.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')

    # predicted_pro is a list
    predicted_pro, pro_mean, ci_range = \
        ensemble_pre_trained_LSTM.predictor_from_sequence(features, t_features, models)
    record = [float(f"{pro_mean:.3f}"), float(f"{ci_range:.3}")]
    record = [t_target, target] + predicted_pro + record  # merge the two lists
    to_be_saved.append(record)


to_be_saved = np.array(to_be_saved)

header_pro = []
for j in np.arange(num_repeate):
    header_pro.append(f"pro{j+1}")

header = f"time_stamps,target,{','.join(header_pro)},pro_mean,pro_95ci_range"
np.savetxt(f"{current_dir}"
           f"Blatten_landslide_predicted_by_{sub_window_size}_{window_overlap}_{trained_model_name}.txt",
           to_be_saved, delimiter=",", fmt="%s", header=header, comments='')
# </editor-fold>


# <editor-fold desc="visualizethe prediction">
# select the short duration for vasulaziton
select_start_time, select_end_time = "2025-05-28T13:15:00", "2025-05-28T13:45:00"
tr = st.trim(starttime=UTCDateTime(select_start_time),
             endtime=UTCDateTime(select_end_time))

date_all = to_be_saved[:, 0]
id_s = np.where(date_all == UTCDateTime(select_start_time).timestamp)[0][0]
id_e = np.where(date_all == UTCDateTime(select_end_time).timestamp)[0][0] + 1

pro_mean, pro_95ci_range = to_be_saved[id_s : id_e, -2], to_be_saved[id_s : id_e, -1]


def psd_plot(ax, ax_twin, st, data_start, data_end, pre_y_pro, ci_range, pre_y_pro_sps, x_interval=0.5):
    start = UTCDateTime(data_start).timestamp
    end = UTCDateTime(data_end).timestamp

    st.trim(UTCDateTime(data_start), UTCDateTime(data_end))
    st.spectrogram(per_lap=0.5, wlen=60, log=False, dbscale=True, mult=True, title="", axes=ax, cmap='inferno')
    ax.images[0].set_clim(-180, -100)

    ax.set_ylim(1, 25)
    ax.set_yticks([1, 10, 20, 25], [1, 10, 20, 25])

    ax.xaxis.set_major_locator(ticker.MultipleLocator(60 * 60 * 2))  # unit is saecond

    x_location = np.arange(start, end + 1, 3600 * x_interval)
    x_ticks = []
    for j, k in enumerate(x_location):
        if j == 0:
            fmt = "%H:%M"
        else:
            fmt = "%H:%M"
        x_ticks.append(datetime.utcfromtimestamp(int(k)).strftime(fmt))

    ax.set_xticks(x_location - start, x_ticks)

    x = np.arange(pre_y_pro.size) / pre_y_pro_sps
    ax_twin.plot(x, pre_y_pro, color="white", lw=1, zorder=2)
    print(pre_y_pro.shape,  np.max(pre_y_pro))
    ax_twin.fill_between(x, pre_y_pro - ci_range, pre_y_pro + ci_range, color="white", alpha=0.5, zorder=2)

    ax_twin.set_ylim(0.05, 1.05)
    ax_twin.set_yticks([0, 0.25, 0.50, 0.75, 1], [0.0, 0.25, 0.50, 0.75, 1.0])

    return ax, ax_twin

pre_y_pro_sps = 1 / (to_be_saved[1, 0] - to_be_saved[0, 0])


pro_st_all = create_trace(to_be_saved[:, -2],
                          UTCDateTime(to_be_saved[0, 0]).strftime("%Y-%m-%dT%H:%M:%S"),
                          pre_y_pro_sps, ref_st=False, return_Trace=False)
pro_st_event = pro_st_all.copy()
pro_st_event.trim(starttime=UTCDateTime(select_start_time),
                  endtime=UTCDateTime(select_end_time))
# make sure the peak is the during the event
benchmark_time = tr[0].stats.starttime + np.argmax(tr[0].data) / tr[0].stats.sampling_rate
benchmark_time = benchmark_time.strftime("%Y-%m-%dT%H:%M:%S")
phi, psi, delta_t, warning_time_float, warning_time_str = inference_matrix(benchmark_time,
                                                                           pro_st_all,
                                                                           pro_st_event,
                                                                           pro_epsilon=0.9)


fig = plt.figure(figsize=(5.5, 4))
gs = gridspec.GridSpec(3, 1, height_ratios=[1, 0.05, 1])

ax = plt.subplot(gs[0])
ax_twin = ax.twinx()
cbar_ax = plt.subplot(gs[1])


ax_twin.axvline(x=(UTCDateTime("2025-05-28T13:24:38")-UTCDateTime(select_start_time)),
                color="green", lw=1, zorder=5)
ax_twin.axvline(x=(UTCDateTime(warning_time_str)-UTCDateTime(select_start_time)),
                color="red", lw=1, zorder=5)

psd_plot(ax, ax_twin, st,
         to_be_saved[id_s, 0],
         to_be_saved[id_e, 0],
         pro_mean,
         pro_95ci_range,
         pre_y_pro_sps,
         x_interval=5/60)

temp = f"{UTCDateTime(start_time).strftime('%Y-%m-%d')},"
ax.set_title(f"Seismic Data Source: {st[0].stats.network}-{st[0].stats.station}-{st[0].stats.channel}",
             weight="bold", fontsize=7)

ax.set_xlabel(f'Time [UTC+0, {temp}]', weight='bold')
ax.set_ylabel('Frequency [Hz]', weight='bold')
ax_twin.set_ylabel("Predicted Pro", weight='bold')


cbar = fig.colorbar(ax.images[0], cax=cbar_ax, orientation="horizontal")
cbar.set_label("Power Spectral Density (dB)")

ax = plt.subplot(gs[2])
ax_twin = ax.twinx()
line1, = ax.plot(output_matrix[id_s:id_e, 2].astype(float), color="C0", label="SNR (max/mean)")
line2, = ax_twin.plot(output_matrix[id_s:id_e, 3].astype(float), color="C1", label="Skewness of PSD")
ax.set_xlim(0, id_e-id_s)
lines = [line1, line2]
labels = [line.get_label() for line in lines]
ax.legend(lines, labels, loc="upper right", fontsize=6)
ax.xaxis.set_major_locator(ticker.MultipleLocator(30))  # unit is saecond

plt.tight_layout()
#plt.savefig(f"{project_root}/docs/"
           #f"Blatten_landslide_psd_{st[0].stats.network}-{st[0].stats.station}-{st[0].stats.channel}.png",
            #dpi=600, transparent=True)
plt.show()
plt.close(fig)

# </editor-fold>

