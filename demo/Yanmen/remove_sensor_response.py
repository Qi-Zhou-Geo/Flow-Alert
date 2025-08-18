#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2025-07-23
# __author__ = Junqing Wang, 202422070321@std.uestc.edu.cn
# __modified by__ = Qi Zhou, qi.zhou@gfz.de, qi.zhou.geo@gmail.com
# Please do not distribute this code without the author's permission


import numpy as np
from obspy import read, Trace, Stream, read_inventory, signal, UTCDateTime


def convert_counts_to_velocity(trace,
                               unit_of_G_v = 'm/s',
                               GAIN_DB = 0,
                               ADC_RANGE_VOLT = 2.5,
                               ADC_BITS = 24,
                               SENSITIVITY = 250):
    '''
    Convert trace.data（unit counts）to ground velocity（unit μm/s or m/s）

    Args:
        trace: Obspy Trace
        unit_of_G_v:
        GAIN_DB:
        ADC_RANGE_VOLT:
        ADC_BITS:
        SENSITIVITY:

    Returns:

    '''

    if not hasattr(trace, 'data') or not isinstance(trace.data, np.ndarray) or trace.data.size == 0:
        print(f"[error] convert_counts_to_velocity: input 'Trace' is empty or invidate。")
        trace.data = np.array([], dtype=np.float32)
        trace.stats.units = unit_of_G_v
        return trace

    try:
        gain = db_to_gain(GAIN_DB)
        amplifier = ground_velocity_amplifier(unit_of_G_v)

        # !!! avoid division by zero or sensitivity to zero
        if SENSITIVITY == 0:
            print("[error] convert_counts_to_velocity: Seneor (SENSITIVITY) can NOT be zero。")
            trace.data = np.zeros_like(trace.data, dtype=np.float32)
            trace.stats.units = unit_of_G_v
            return trace

        v_per_count = ADC_RANGE_VOLT / (2 ** (ADC_BITS - 1)) / gain
        mps_per_count = v_per_count / SENSITIVITY
        trace.data = trace.data * mps_per_count * amplifier
        trace.stats.units = unit_of_G_v
    except Exception as e:
        print(f"[error] convert_counts_to_velocity: Convert 'counts' to ground velocity: {e}")
        # !!! set the data to zero or leave it as is to avoid propagating invalid data
        trace.data = np.zeros_like(trace.data, dtype=np.float32)
        # still sets the unit, indicating that conversion was attempted
        trace.stats.units = unit_of_G_v

    return trace

def db_to_gain(db):
    try:
        return 10 ** (db / 20)
    except Exception as e:
        print(f"[error] db_to_gain: Convert 'dB' to gain ({db} dB): {e}")
        return 1.0


def ground_velocity_amplifier(unit_of_G_v):
    '''
    Retuen the amplifier for the physical ground velocity

    Args:
        unit_of_G_v: str, either 'um/s' or 'm/s'

    Returns:
        amplifier: float

    '''
    if unit_of_G_v == 'um/s':
        amplifier = 1e6
    elif unit_of_G_v == 'm/s':
        amplifier = 1e2 # this is not correct
    else:
        print(f"[error] ground_velocity_amplifier: "
              f"pleae check your unit_of_G_v {unit_of_G_v}: {e}")

    return amplifier


def seismic_data_processing(data_path, data_name, f_min=1, f_max=45):

    st = read(f"{data_path}/{data_name}")
    st.merge(method=1, fill_value='latest', interpolation_samples=0)
    st._cleanup()
    st.detrend('linear')
    st.detrend('demean')

    st = convert_counts_to_velocity(trace=st[0])
    st = Stream(st)
    st.filter("bandpass", freqmin=f_min, freqmax=f_max)
    st.detrend('linear')
    st.detrend('demean')

    station_id = {'5200184':"STA01",
                  '5200202':"STA02",
                  '5200206':"STA03",
                  '5200183':"STA04",
                  '5200194':"STA05",
                  '5200207':"STA06",
                  '5200201':"STA07",
                  '5200182':"STA08"}

    st[0].stats.station = station_id.get(st[0].stats.station)

    return st
