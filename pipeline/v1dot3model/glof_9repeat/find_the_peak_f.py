#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2025-02-17
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission


import os
import yaml
import argparse

import pandas as pd
import numpy as np


import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

from obspy import UTCDateTime, read

# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path
current_dir = Path(__file__).resolve().parent

# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent.parent
import sys
sys.path.append(str(project_root))
# </editor-fold>

# import the custom functions
from functions.seismic.seismic_data_processing import load_seismic_signal
from functions.seismic.welch_spectrum import welch_psd

plt.rcParams.update({'font.size': 7,
                     'font.family': "Arial",
                     'axes.formatter.limits': (-4, 6),
                     'axes.formatter.use_mathtext': True})

# from upstream to downstream
seismic_sta_list = ["NEP08", "NEP07", "NEP06", "NEP10", "NEP05", "NEP04"]

# <editor-fold desc="prepare data">
idx_event = 85 - 1
default_data_path = f"{project_root}/config/data_path.yaml"
with open(default_data_path, "r") as f:
    config = yaml.safe_load(f)
    sac_path = config[f"glic_sac_dir"]
    event_catalog_version = config[f"event_catalog_version"]
    print(f"event_catalog_version: {event_catalog_version}")

file_path = f"{project_root}/data/event_catalog/{event_catalog_version}"
df = pd.read_csv(f"{file_path}", header=0)

row_idx = df.loc[idx_event]  # select row_idx
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

def archive_data(seismic_sta_list,
                 catchment="Bothekoshi",
                 seismic_network="XN",
                 component="HHZ"):

    for idx, sta in enumerate(seismic_sta_list):
        data_start = "2016-07-05T06:00:00"
        data_end = "2016-07-06T06:00:00"

        st = load_seismic_signal(catchment,
                                 seismic_network,
                                 sta,
                                 component,
                                 data_start,
                                 data_end,
                                 f_min=1,
                                 f_max=45,
                                 remove_sensor_response=True,
                                 raw_data=False)
        print(st[0].stats)
        os.makedirs(f"{current_dir}/seismic_data", exist_ok=True)
        st.write(f"{current_dir}/seismic_data"
                 f"/Bothekoshi-XN-{sta}-HHZ-1-45-2016-07-05T06:00:00-2016-07-06T06:00:00.mseed",
                 format="MSEED")

# archive_data(seismic_sta_list=seismic_sta_list)


fig = plt.figure(figsize=(5, 5))
gs = gridspec.GridSpec(1, 1)
ax = plt.subplot(gs[0])

for idx, sta in enumerate(seismic_sta_list):
    st = read(f"{current_dir}/seismic_data"
              f"/Bothekoshi-XN-{sta}-HHZ-1-45-2016-07-05T06:00:00-2016-07-06T06:00:00.mseed")
    # st.plot()
    st.trim(UTCDateTime(sta_s), UTCDateTime(sta_e))

    data = st[0].data
    sampling_rate = st[0].stats.sampling_rate

    f_min, f_max = 1, 45
    freq, psd, psd_unit = welch_psd(data, sampling_rate,
                                    f_min, f_max,
                                    segment_window=10,
                                    scaling="density",
                                    unit_dB=True)
    max_psd_id = np.argmax(psd)
    peak_frequency = freq[max_psd_id]

    print(f"{sta}, {peak_frequency}, {np.max(psd)}")
    ax.plot(freq, psd, label=f"{sta}")

ax.set_ylim(-200, -50)
ax.legend(fontsize=6)
ax.set_xscale("log")
ax.set_ylabel(f"Power Spectral Density [PSD]", weight='bold')
ax.set_xlabel('Frequemcy [Hz]', weight='bold')
plt.tight_layout()
plt.savefig(f"{current_dir}/GLOF_psd.png", dpi=600, transparent=True)
plt.show()
plt.close(fig=fig)
