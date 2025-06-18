#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-12-27
#__author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission

import os
import argparse

import pytz
from datetime import datetime


import numpy as np
import pandas as pd
from itertools import combinations

import scipy
from scipy.signal import hilbert, lfilter, butter, spectrogram, coherence, correlate, correlation_lags
from scipy.stats import kurtosis, skew, iqr, wasserstein_distance

from obspy import read, Stream, read_inventory, signal
from obspy.core import UTCDateTime # default is UTC+0 time zone


# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path
current_dir = Path(__file__).resolve().parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent
import sys
sys.path.append(str(project_root))
# </editor-fold>


# import the custom functions
from calculate_features.define_path import check_folder

def load_st_as_npy(input_year, station_list, catchment_name, seismic_network, input_component, julday):

    julday = str(julday).zfill(3)

    temp = []
    for idx, input_station in enumerate(station_list):

        folder_path_txt, folder_path_npy, folder_path_net = check_folder(catchment_name, seismic_network, input_year, input_station, input_component)

        loaded = np.load(f"{folder_path_npy}/{input_year}_{input_station}_{input_component}_{julday}.npz", allow_pickle=True)
        sps = float(loaded['sampling_rate']) # HZ
        data = loaded['data'] # default, meter per second
        #input_window_size = loaded['window_size'] # same as input window size
        temp.append(data)

    min_len = min(arr.shape[0] for arr in temp)
    temp = [arr[:min_len] for arr in temp] # trim all arrays to the same length

    temp = np.vstack(temp, dtype=float).T

    return sps, temp

def rms_network(waveform_segment, epsilon=1.0):
    '''

    Args:
        waveform_segment: 2D numpy array, each column represents the seismic signals for one station
        epsilon: int or float, default value when "ZeroDivisionError"

    Returns:
        float value
    '''

    rms_per_trace = np.sqrt(np.mean(waveform_segment ** 2, axis=0))
    min_rms_idx = np.argmin(rms_per_trace)
    max_rms_idx = np.argmax(rms_per_trace)

    try:
        ratio_rms = np.max(rms_per_trace) / np.min(rms_per_trace)
    except ZeroDivisionError:
        ratio_rms = epsilon

    q1 = np.percentile(waveform_segment, 25, axis=0)
    q3 = np.percentile(waveform_segment, 75, axis=0)
    iqr_per_trace = q3 - q1

    try:
        ratio_iqr = np.max(iqr_per_trace) / np.min(iqr_per_trace)
    except ZeroDivisionError:
        ratio_iqr = epsilon

    return min_rms_idx, max_rms_idx, ratio_rms, ratio_iqr

def Coher(trace1, trace2):  # cross-coherence from Gosia

    f = trace1 ** 2 / np.sum(trace1 ** 2)
    g = trace2 ** 2 / np.sum(trace2 ** 2)
    wd = wasserstein_distance(f, g)

    lentrace = len(trace1)
    maxlag = lentrace
    goodnumber = int(2 ** (np.ceil(np.log2(lentrace)) + 2))

    tr2 = np.zeros(goodnumber)
    tr2[0:lentrace] = trace2
    tr2 = scipy.fftpack.fft(tr2, overwrite_x=True)
    tr2.imag *= -1
    tr1 = np.zeros(goodnumber)
    tr1[maxlag:maxlag + lentrace] = trace1
    tr1 = scipy.fftpack.fft(tr1, overwrite_x=True)

    try:
        tr_cc = (tr1 * tr2) / (np.absolute(tr1) * np.absolute(tr2))
    except:# Calculate the tr_cc array, handling invalid values
        tr_cc = np.divide(tr1 * tr2, np.absolute(tr1) * np.absolute(tr2), out=np.zeros_like(tr1), where=(tr1 != 0) & (tr2 != 0) )
        print("method 2")

    tr_cc[np.isnan(tr2)] = 0.0
    tr_cc[np.isinf(tr2)] = 0.0

    go = scipy.fftpack.ifft(tr_cc, overwrite_x=True)[0:2 * maxlag + 1].real

    coherence_values = np.max(go)
    lagTime = np.argmax(go)

    return wd, lagTime, coherence_values


def record_data_header(input_year, input_component, julday, folder_path_net):

    feature_names = ['time_window_start', 'time_stamps', 'component',
                     'id_maxRMS', 'id_minRMS',
                     'ration_maxTOminRMS', 'ration_maxTOminIQR',
                     'mean_coherenceOfNet', 'max_coherenceOfNet',
                     'mean_lagTimeOfNet', 'std_lagTimeOfNet',
                     'mean_wdOfNet', 'std_wdOfNet']  # timestamps + component + 9 network features

    feature_names = np.array(feature_names)


    # give features title and be careful the file name and path
    with open(f"{folder_path_net}/{input_year}_{input_component}_{julday}_net.txt", 'w') as file:
        np.savetxt(file, [feature_names], header='', delimiter=',', comments='', fmt='%s')

def run_cal_loop(catchment_name, seismic_network, input_year, station_list, input_component, input_window_size, id1, id2, folder_path_net):

    for julday in range(id1, id2):  # 91 = 1st of May to 305=31 of Nov.
        d = UTCDateTime(year=input_year, julday=julday)  # the start day, e.g.2014-07-12 00:00:00
        julday = str(julday).zfill(3)

        # write the seismic features header
        record_data_header(input_year, input_component, julday, folder_path_net)

        # load data-60s as 2D array, each column represents one station
        sps, waveform_array = load_st_as_npy(input_year, station_list, catchment_name, seismic_network, input_component, julday)
        arra_length = int(waveform_array.shape[0] / (sps * input_window_size)) # return as minutes

        for step in range(arra_length):

            # processed 1-min seismic data-60s array
            waveform_segment = waveform_array[int(step*sps*input_window_size): int((step+1)*sps*input_window_size), :]
            # first three RMS related network features
            id_max, id_min, ratio_rms, ratio_iqr = rms_network(waveform_segment)

            # create combinations of columns (pairs of indices, NOT data-60s)
            pair_combinations = combinations(range(waveform_segment.shape[1]), 2) # from [a,b,c] to (a,b), (a,c), (b,c)

            wd_list, lagTime_list, coherence_list = [], [], []
            for pair1, pair2 in pair_combinations:
                trace1 = waveform_segment[:, pair1]
                trace2 = waveform_segment[:, pair2]
                xcorr = Coher(trace1, trace2)  # Use Gosia's code

                wd_list.append(xcorr[0])
                lagTime_list.append(xcorr[1])
                coherence_list.append(np.max(xcorr[2]))  # only select the max coherence values between two stations

            time_stamps = float(d + (step) * input_window_size)
            time = datetime.fromtimestamp(time_stamps, tz=pytz.utc)
            time = time.strftime('%Y-%m-%dT%H:%M:%S')

            mean_coherence = np.mean(coherence_list)  # coherence related network features
            max_coherence = np.max(coherence_list)

            mean_lagTime = np.mean(lagTime_list)  # lagTime related network features
            std_lagTime = np.std(lagTime_list)

            mean_wd = np.mean(wd_list)  # wd related network features
            std_wd = np.std(wd_list)

            arr = np.array((time, time_stamps, input_component,
                            id_max, id_min,
                            ratio_rms, ratio_iqr,
                            mean_coherence, max_coherence,
                            mean_lagTime, std_lagTime,
                            mean_wd, std_wd))
            # do not give header here ( see [feature_names] )
            with open(f"{folder_path_net}/{input_year}_{input_component}_{julday}_net.txt", 'a') as file:
                np.savetxt(file, arr.reshape(1, -1), header='', delimiter=',', comments='', fmt='%s')#fmt_list)


def main(catchment_name, seismic_network, input_year, station_list, input_component, input_window_size, id):  # Update the global variables with the values from command-line arguments
    print(f"Start Job: {input_year}, {input_component}: ", datetime.now().strftime("%Y-%m-%dT%H:%M:%S") )

    # check the folder
    try:
        folder_path_txt, folder_path_npy, folder_path_net = check_folder(catchment_name, seismic_network, input_year, None, input_component)
    except FileExistsError as e:
        print(f"{input_year}, {input_component}, {input_window_size}, {id}, \n"
              f"Exception {e}: Directory already exists, ignoring.")
    except Exception as e:
        print(f"{input_year}, {input_component}, {input_window_size}, {id}, \n"
              f"Exception {e}")

    id1, id2 = id, id + 1
    run_cal_loop(catchment_name, seismic_network, input_year, station_list, input_component, input_window_size, id1, id2, folder_path_net)


    print(f"End Job: {input_year}, {input_component}: ", datetime.now().strftime("%Y-%m-%dT%H:%M:%S") )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--catchment_name", type=str, default="Illgraben", help="check the year")
    parser.add_argument("--seismic_network", type=str, default="9S", help="check the year")
    parser.add_argument("--input_year", type=int, default=2020, help="cal_year")
    parser.add_argument("--station_list", nargs='+', type=str, help="list of stations")
    parser.add_argument("--input_component", type=str, default="ILL12", help="check the input_station")
    parser.add_argument("--input_window_size", type=int, default=60, help="check the calculate window size")
    parser.add_argument("--id", type=int, default=60, help="check the calculate window size")
    args = parser.parse_args()

    main(args.catchment_name, args.seismic_network,
         args.input_year, args.station_list, args.input_component, args.input_window_size, args.id)
