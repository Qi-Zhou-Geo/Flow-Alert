#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2025-07-23
# __author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this functions without the author's permission

import yaml
import os

import numpy as np
import pandas as pd

from pathlib import Path

from obspy import read, Trace, Stream, read_inventory, signal
from obspy.core import UTCDateTime  # default is UTC+0 time zone
from obspy.signal.invsim import simulate_seismometer

# for obspy taper
import scipy.signal
import scipy.signal.windows

scipy.signal.hann = scipy.signal.windows.hann

# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path

current_dir = Path(__file__).resolve().parent
# using ".parent" on "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys

sys.path.append(str(project_root))
# </editor-fold>


def remove_LD_STA01_response(trace, paz, pre_filt=(0.5, 1.0, 45.0, 50.0)):
    '''
    Remove sensor response for LD STA01 station
    This is based on the description of Prof. Dr. Dan Wang

    Args:
        trace: Obspy trace

    Returns:
        trace: Obspy trace
    '''

    gain_db = 18  # unit is dB
    linear_gain = 10 ** (gain_db / 20)  # 18 dB gain approax. 7.94

    trace.data = trace.data / linear_gain  # remove linear gain
    trace.data = trace.data / 1000.0  # mV to Voltage

    # we are interest freq. in 1-45 Hz
    trace.simulate(paz_remove=paz,
                   paz_simulate=None,
                   remove_sensitivity=True,
                   pre_filt=pre_filt)

    trace = Stream(trace)

    return trace


def config_snesor_parameter(catchment_name, seismic_network):

    default_data_path = f"{project_root}/config/data_path.yaml"
    with open(default_data_path, "r") as f:
        config = yaml.safe_load(f)
        sac_path = config[f"glic_sac_dir"]
        feature_path = config[f"seismic_feature_dir"]
        event_catalog_version = config[f"event_catalog_version"]
        print(f"event_catalog_version: {event_catalog_version}")

    file_path = f"{project_root}/data/event_catalog/{event_catalog_version}"
    df = pd.read_csv(f"{file_path}", header=0)

    # select the first folumn
    idx = df[(df["Catchment"] == catchment_name) & (df["Network"] == seismic_network)].index[0]
    row_idx = df.loc[idx]  # select row_idx
    response_type, sensor_type = row_idx["Response-type"], row_idx["Sensor-Logger-type"]

    sac_path = f"{sac_path}/{row_idx['Continent']}/{catchment_name}"
    feature_path = f"{feature_path}/{row_idx['Continent']}/{catchment_name}"

    return sac_path, feature_path, response_type, sensor_type


def define_normalization_factor(catchment_name, seismic_network):
    if catchment_name == "Ruapehu" and seismic_network == "MR":
        # the amplitude (unit by counts) divide 2 * 10**9 to velocity meter/second
        normalization_factor = 5 * 10 ** 9
    elif catchment_name == "Goulinping" and seismic_network == "CC":
        # the amplitude (unit by counts) divide 2 * 10**9 to velocity meter/second
        normalization_factor = 2 * 10 ** 9
    else:
        print(f"Error in <define_normalization_factor>. \n"
              f"Please check the 'catchment_name, seismic_networ' {catchment_name, seismic_network}")

    return normalization_factor


def manually_remove_sensor_response(trace, sensor_type, pre_filt=(0.5, 1.0, 45.0, 50.0)):
    '''
    Manually remove the sensor response

    Args:
        trace: Obspy Stream or trace: seismic stream that deconvolved, make sure the stream only hase one trace
        sensor_type: sensor type

    Returns:
        st (obspy.core.stream): seismic stream that removed the sensor response
    '''

    # <editor-fold desc="Reference">
    # https://www.gfz-potsdam.de/en/section/geophysical-imaging/infrastructure/geophysical-instrument-pool-potsdam-gipp/pool-components/clipp-werte
    # https://www.gfz-potsdam.de/en/section/geophysical-imaging/infrastructure/geophysical-instrument-pool-potsdam-gipp/pool-components/poles-and-zeros/trillium-c-120s

    # total sensitivity = sensitivity_sensor * sensitivity_logger
    # total sensitivity: counts per (meter per second) [counts / (m/s)]
    # sensitivity_sensor: volts per (meter per second) [V·s/m, or V/(m/s)]
    # sensitivity_logger: counts per volts [counts / V]

    # Zeros and Poles for sensor
    # GFZ geophone
    # https://www.gfz.de/en/section/geophysical-imaging/infrastructure/geophysical-instrument-pool-potsdam-gipp/pool-components/poles-and-zeros/3d-geophone
    # Trillium Compact; 120 s ... 108 Hz
    # https://www.gfz.de/en/section/geophysical-imaging/infrastructure/geophysical-instrument-pool-potsdam-gipp/pool-components/poles-and-zeros/trillium-c-120s

    # Zeros and Poles for logger
    # DATA-CUBE3
    # https://www.gfz.de/en/section/geophysical-imaging/infrastructure/geophysical-instrument-pool-potsdam-gipp/pool-components/clipp-werte

    # sensor_logger_type = {
    # works for AA
    #     'zeros': [(+ bj)],
    #
    #     'poles': [(+ bj)],

    # alway set as 1 for Obspy, and set sensitivity as: total sensitivity = sensitivity_sensor * sensitivity_logger
    #     'gain': 1,
    #     'sensitivity': sensitivity_sensor * sensitivity_logger # ctotal sensitivity [counts / (m/s)]
    # }

    # </editor-fold>

    # <editor-fold desc="PAZ">
    paz_IGU_16HR_EB_3C_5Hz = {
        # works for SmartSolo IGU-16HR 3C
        # works for Jiangjia 2023 -> Prof. Dr. Shuai Li 2025 data, Luding 2023 STA01 DP* data
        'zeros': [(0 + 0j),
                  (0 + 0j)],

        'poles': [(-22.211059 + 22.217768j),
                  (-22.211059 - 22.217768j)],

        'gain': 1000,
        'sensitivity': 76.7  # counts / (m/s)
    }

    paz_3D_Geophone_PE_6_B16 = {
        # works for 3D Geophone PE-6/B; 4.5 ... 500 Hz(*) with DATA-CUBE3,
        # works for Jiangjia (gain 16) -> Prof. Dr. Shuai Li 2023 data
        'zeros': [(0 + 0j),
                  (0 + 0j)],

        'poles': [(-19.78 + 20.20j),
                  (-19.78 - 20.20j)],

        'gain': 1,
        'sensitivity': 27.7 * 6.5574 * 1e7 # counts / (m/s)
    }

    paz_3D_Geophone_PE_6_B32 = {
        # works for 3D Geophone PE-6/B; 4.5 ... 500 Hz(*) with DATA-CUBE3,
        # works for Ergou (gain 32) -> Prof. Dr. Yan Yan 2022 data
        'zeros': [(0 + 0j),
                  (0 + 0j)],

        'poles': [(-19.78 + 20.20j),
                  (-19.78 - 20.20j)],

        'gain': 1,
        'sensitivity': 27.7 * 1.3115 * 1e8 # counts / (m/s)
    }

    paz_3D_NoiseScope = {
        # works for paz_3D_NoiseScope
        # works for Foutangba -> Prof. Dr. Yan Yan 2022 data

        'zeros': [(0 + 0j),
                  (0 + 0j)],

        'poles': [(-0.444221 - 0.6565j),
                  (-0.444221 + 0.6565j),
                  (-222.110595 - 222.17759j),
                  (-222.110595 + 222.17759j)],

        'gain': 298,
        # 6.71140939 * 1e9 = 2000 [V·s/m] * 1/298 [nV/count, nanovolts/count]
        'sensitivity': 6.71140939 * 1e9  # counts / (m/s)
    }

    paz_PMS10 = {
        # works for PMS-10 short-period seismometer,
        # works for Tianmo data
        'zeros': [(0 + 0j),
                  (0 + 0j)],

        'poles': [(-0.44429 + 0.44429j),
                  (-0.44429 - 0.44429j),
                  (-666.43 + 666.43j),
                  (-666.43 - 666.43j)],

        'gain': 888264, # MUST set as 888264
        'sensitivity': 8.38 * 1e8  # counts / (m/s)
    }
    # </editor-fold>

    # <editor-fold desc="Make copy">
    corrected_trace = trace.copy()
    if isinstance(corrected_trace, Stream):
        corrected_trace.merge(method=1, fill_value='latest', interpolation_samples=0)
        corrected_trace = corrected_trace[0]
    elif isinstance(corrected_trace, Trace):
        pass
    else:
        print(f"!!! Error\n"
              f"Make sure the input for <manually_remove_sensor_response> is Obspy 'Trace' or 'Stream'.")
    # </editor-fold>

    if sensor_type == "IGU_16HR_EB_3C_5Hz":
        paz = paz_IGU_16HR_EB_3C_5Hz
    elif sensor_type == "paz_3D_Geophone_PE_6_B16":
        paz = paz_3D_Geophone_PE_6_B16
    elif sensor_type == "paz_3D_Geophone_PE_6_B32":
        paz = paz_3D_Geophone_PE_6_B32
    elif sensor_type == "paz_3D_NoiseScope":
        paz = paz_3D_NoiseScope
    elif sensor_type == "paz_PMS10":
        paz = paz_PMS10
    else:
        print(f"please check the sensor_type: {sensor_type}")

    corrected_trace.simulate(
        paz_remove=paz,
        paz_simulate=None,
        remove_sensitivity=True,
        pre_filt=pre_filt
    )

    corrected_trace = Stream(corrected_trace)

    return corrected_trace


def load_seismic_signal(catchment_name, seismic_network, station, component,
                        data_start, data_end,
                        f_min=1, f_max=45,
                        remove_sensor_response=True,
                        raw_data=False):

    d1 = UTCDateTime(data_start)
    d2 = UTCDateTime(data_end)

    # config the snesor parameter based on seismci network code
    sac_path, feature_path, response_type, sensor_type = config_snesor_parameter(catchment_name, seismic_network)

    # make sure all you file is structured like this
    # '~/sac_path/continent_name/catchment_name/year/station/componment/
    #   seismic_network.station.componment.year.julday.mseed'
    # note the julday like: str(d1.julday).zfill(3)

    file_dir = f"{sac_path}/{d1.year}/{station}/{component}/"

    if d1.julday == d2.julday:
        data_name = f"{seismic_network}.{station}.{component}.{d1.year}.{str(d1.julday).zfill(3)}.mseed"
        st = read(file_dir + data_name)
    else:
        st = Stream()
        for n in np.arange(d1.julday - 1, d2.julday + 1):
            data_name = f"{seismic_network}.{station}.{component}.{d1.year}.{str(n).zfill(3)}.mseed"
            st += read(file_dir + data_name)

        # to avoid the "edge" effect in both sides
        st.trim(starttime=d1 - 3600 * 6,
                endtime=d2 + 3600 * 6,
                nearest_sample=False)

    if raw_data is True:
        return st

    st.merge(method=1, fill_value='latest', interpolation_samples=0)
    st._cleanup()
    st.detrend('linear')
    st.detrend('demean')
    st.taper(max_percentage=0.05)  # to avoid the "edge" effect in both sides

    if remove_sensor_response is True:

        if response_type == "xml":  # with xml file
            meta_file = [f for f in os.listdir(f"{sac_path}/meta_data") if f.startswith(seismic_network)][0]
            inv = read_inventory(f"{sac_path}/meta_data/{meta_file}")
            st.remove_response(inventory=inv)
        elif response_type == "simulate":  # with poles and zeros
            st = manually_remove_sensor_response(st, sensor_type)
        elif response_type == "manually":  # without poles and zeros
            normalization_factor = define_normalization_factor(catchment_name, seismic_network)
            st[0].data = st[0].data / normalization_factor
        elif response_type == "do_not_need":  # only for Chalk_Cliffs 2014 data
            pass
        else:
            print(f"please check the response_type: {response_type}")

        st = Stream(st)

        if st[0].stats.sampling_rate <= 2 * f_max:
            st.filter("highpass", freq=f_min, zerophase=True)
        else:
            st.filter("bandpass", freqmin=f_min, freqmax=f_max)

        st.trim(starttime=d1, endtime=d2, nearest_sample=False)
        st.detrend('linear')
        st.detrend('demean')
        # st.taper(max_percentage=0.05, type='hann')

    else:
        st.trim(starttime=d1, endtime=d2, nearest_sample=False)

    return st

def load_seismic_pieces(catchment_name, seismic_network, station, component,
                        data_start, data_end,
                        f_min=1, f_max=45,
                        remove_sensor_response=True,
                        raw_data=False):

    d1 = UTCDateTime(data_start)
    d2 = UTCDateTime(data_end)

    # config the snesor parameter based on seismci network code
    sac_path, feature_path, response_type, sensor_type = config_snesor_parameter(catchment_name, seismic_network)

    # make sure all you file is structured like this
    # '~/sac_path/continent_name/catchment_name/year/station/componment/
    #   seismic_network.station.componment.year.julday.mseed'
    # note the julday like: str(d1.julday).zfill(3)

    file_dir = f"{sac_path}/{d1.year}/{station}/{component}/"

    if d1.julday == d2.julday:
        data_name = f"{seismic_network}.{station}.{component}.{d1.year}.{str(d1.julday).zfill(3)}.mseed"
        st = read(file_dir + data_name)
    else:
        st = Stream()
        for n in np.arange(d1.julday, d2.julday + 1):
            data_name = f"{seismic_network}.{station}.{component}.{d1.year}.{str(n).zfill(3)}.mseed"
            try:
                st += read(file_dir + data_name)
            except FileNotFoundError:
                print(f"please check the data: {data_name}")

    if raw_data is True:
        return st

    st.merge(method=1, fill_value='latest', interpolation_samples=0)
    st._cleanup()
    st.detrend('linear')
    st.detrend('demean')
    st.taper(max_percentage=0.05)  # to avoid the "edge" effect in both sides

    if remove_sensor_response is True:

        if response_type == "xml":  # with xml file
            meta_file = [f for f in os.listdir(f"{sac_path}/meta_data") if f.startswith(seismic_network)][0]
            inv = read_inventory(f"{sac_path}/meta_data/{meta_file}")
            st.remove_response(inventory=inv)
        elif response_type == "simulate":  # with poles and zeros
            st = manually_remove_sensor_response(st, sensor_type)
        elif response_type == "manually":  # without poles and zeros
            normalization_factor = define_normalization_factor(catchment_name, seismic_network)
            st[0].data = st[0].data / normalization_factor
        elif response_type == "do_not_need":  # only for Chalk_Cliffs 2014 data
            pass
        else:
            print(f"please check the response_type: {response_type}")

        st = Stream(st)

        if st[0].stats.sampling_rate <= 2 * f_max:
            st.filter("highpass", freq=f_min, zerophase=True)
        else:
            st.filter("bandpass", freqmin=f_min, freqmax=f_max)

        st.detrend('linear')
        st.detrend('demean')
        # st.taper(max_percentage=0.05, type='hann')

    else:
        pass

    start_time = st[0].stats.starttime
    end_time = st[0].stats.endtime
    t1 = UTCDateTime(year=start_time.year, month=start_time.month, day=start_time.day, hour=start_time.hour)
    t1 = t1 + (start_time.minute + 1) * 60
    t2 = UTCDateTime(year=end_time.year, month=end_time.month, day=end_time.day, hour=end_time.hour)
    t2 = t2 + (end_time.minute - 1) * 60
    st = st.trim(t1, t2)

    return st
