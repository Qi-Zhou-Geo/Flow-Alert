#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2025-02-07
#__author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
#__find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission

import yaml
import os
import numpy as np

from pathlib import Path

from obspy import read, Stream, read_inventory, signal
from obspy.core import UTCDateTime # default is UTC+0 time zone
from obspy.signal.invsim import simulate_seismometer

def config_snesor_parameter(catchment_name, seismic_network):

    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent

    config_path = f"{project_root}/config/config_I-O.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        glic_sac_dir = config[f"glic_sac_dir"]

    config_path = f"{project_root}/config/config_catchment_code.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)


    config = config[f"{catchment_name}-{seismic_network}"]
    path_mapping = config["path_mapping"]
    sac_path = Path(glic_sac_dir) / path_mapping
    response_type = config["response_type"]
    sensor_type = config["sensor_type"]
    normalization_factor = config["normalization_factor"]

    return sac_path, response_type, sensor_type, normalization_factor


def load_seismic_signal(catchment_name, seismic_network, station, component, data_start, data_end,
                        f_min=1, f_max=45, remove_sensor_response=True, raw_data=False):

    d1 = UTCDateTime(data_start)
    d2 = UTCDateTime(data_end)


    # config the snesor parameter based on seismci network code
    sac_path, response_type, sensor_type, normalization_factor = config_snesor_parameter(catchment_name, seismic_network)

    # make sure all you file is structured like this
    # '/storage/vast-gfz-hpc-01/project/seismic_data_qi/seismic/continent_name/catchment_name'
    file_dir = f"{sac_path}/{d1.year}/{station}/{component}/"

    if d1.julday == d2.julday:
        data_name = f"{seismic_network}.{station}.{component}.{d1.year}.{str(d1.julday).zfill(3)}.mseed"
        st = read(file_dir + data_name)
    else:
        st = Stream()
        for n in np.arange(d1.julday, d2.julday+1):
            data_name = f"{seismic_network}.{station}.{component}.{d1.year}.{str(n).zfill(3)}.mseed"
            st += read(file_dir + data_name)

    if raw_data is True:
        return st

    st.merge(method=1, fill_value='latest', interpolation_samples=0)
    st._cleanup()
    st.detrend('linear')
    st.detrend('demean')

    if remove_sensor_response is True:
        if response_type == "xml": # with xml file
            meta_file = [f for f in os.listdir(f"{sac_path}/meta_data") if f.startswith(seismic_network)][0]
            inv = read_inventory(f"{sac_path}/meta_data/{meta_file}")
            st.remove_response(inventory=inv)
        else:
            print(f"please check the response_type: {response_type}")

        if st[0].stats.sampling_rate <= 2 * f_max:
            st.filter("highpass", freq=f_min, zerophase=True)
        else:
            st.filter("bandpass", freqmin=f_min, freqmax=f_max)

        st.trim(starttime=d1, endtime=d2, nearest_sample=False)
        st.detrend('linear')
        st.detrend('demean')

    else:
        st.trim(starttime=d1, endtime=d2, nearest_sample=False)

    return st

