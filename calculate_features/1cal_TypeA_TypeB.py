#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-11-02
#__author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission

import os
import argparse

import pytz
from datetime import datetime

import numpy as np
import pandas as pd

from scipy.signal import hilbert, lfilter, butter, spectrogram
from scipy.stats import kurtosis, skew, iqr

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
# import Qi's all features (by *)
from calculate_features.Type_A_features import calBL_feature
# import Clement's all features (by *)
from calculate_features.Type_B_features import calculate_all_attributes
from functions.seismic.seismic_data_processing import load_seismic_signal


def cal_attributes_B(data_array, sps): # the main function is from Clement
    # sps: sampling frequency; flag=0: one component seismic signal;
    features = calculate_all_attributes(Data=data_array, sps=sps, flag=0)[0] # feature 1 to 60
    feature_array = features[1:]# leave features[0]=event_duration

    return feature_array # 59 features

def cal_attributes_A(data_array, scaling=1e9, ruler=300): # the main function is from Qi
    data_array_nm = data_array * scaling # converty m/s to nm/s
    feature_array = calBL_feature(data_array_nm, ruler)

    return feature_array # 17 features


def record_data_header(input_year, input_station, input_component, julday, folder_path_txt):

    featureName1 = ['time_window_start', 'time_stamps', 'station', 'component',
                    'RappMaxMean', 'RappMaxMedian', 'AsDec', 'KurtoSig','KurtoEnv', 'SkewnessSig','SkewnessEnv',
                    'CorPeakNumber', 'INT1', 'INT2', 'INT_RATIO', 'ES_0', 'ES_1', 'ES_2', 'ES_3', 'ES_4', 'KurtoF_0',
                    'KurtoF_1', 'KurtoF_2', 'KurtoF_3', 'KurtoF_4', 'DistDecAmpEnv','env_max_to_duration', 'RMS', 'IQR', 'MeanFFT', 'MaxFFT',
                    'FmaxFFT', 'FCentroid', 'Fquart1', 'Fquart3', 'MedianFFT', 'VarFFT', 'NpeakFFT', 'MeanPeaksFFT', 'E1FFT','E2FFT','E3FFT', 'E4FFT',
                    'gamma1', 'gamma2', 'gammas', 'SpecKurtoMaxEnv', 'SpecKurtoMedianEnv', 'RatioEnvSpecMaxMean', 'RatioEnvSpecMaxMedian','DistMaxMean',
                    'DistMaxMedian', 'NbrPeakMax', 'NbrPeakMean', 'NbrPeakMedian','RatioNbrPeakMaxMean', 'RatioNbrPeakMaxMedian',
                    'NbrPeakFreqCenter', 'NbrPeakFreqMax', 'RatioNbrFreqPeaks','DistQ2Q1', 'DistQ3Q2', 'DistQ3Q1'] # RF features

    featureName2 = ['time_window_start', 'time_stamps', 'station', 'component',
                    'digit1','digit2','digit3','digit4','digit5', 'digit6','digit7','digit8','digit9',
                    'max', 'goodness', 'iqr', 'magnitude_range', 'alpha', 'ks', 'MannWhitneU', 'follow'] # BL features

    featureName1, featureName2 = np.array(featureName1), np.array(featureName2)

    # give features title and be careful the file name and path
    with open(f"{folder_path_txt}/{input_year}_{input_station}_{input_component}_{julday}_B.txt", 'a') as file1:
        np.savetxt(file1, [featureName1], header='', delimiter=',', comments='', fmt='%s')
    # give features title and be careful the file name and path
    with open(f"{folder_path_txt}/{input_year}_{input_station}_{input_component}_{julday}_A.txt", 'a') as file2:
        np.savetxt(file2, [featureName2], header='', delimiter=',', comments='', fmt='%s')


def record_data(input_year, input_station, input_component, arr, feature_type, julday, folder_path_txt):

    arr = arr.reshape(1, -1)
    # seismic features to be saved
    with open(f"{folder_path_txt}/{input_year}_{input_station}_{input_component}_{julday}_{feature_type}.txt", 'a') as file:
        np.savetxt(file, arr, header='', delimiter=',', comments='', fmt='%s')  # '%.4f')


def loop_time_step(st, input_year, input_station, input_component, input_window_size, julday, folder_path_txt):

    # columns array to store the seismic data-60s for network features
    sps = int(st[0].stats.sampling_rate)
    d = UTCDateTime(year=input_year, julday=julday)  # the start day, e.g.2014-07-12 00:00:00

    for step in range(0, int(3600 * 24 / input_window_size)):  # 1440 = 24h * 60min
        d1 = d + (step) * input_window_size  # from minute 0, 1, 2, 3, 4
        d2 = d + (step + 1) * input_window_size  # from minute 1, 2, 3, 4, 5

        # float timestamps, mapping station and component
        time = datetime.fromtimestamp(d1.timestamp, tz=pytz.utc)
        time = time.strftime("%Y-%m-%dT%H:%M:%S")
        id = np.array([time, d1.timestamp, input_station, input_component])

        tr = st.copy()
        tr.trim(starttime=d1, endtime=d2, nearest_sample=False)
        seismic_data = tr[0].data[:sps * input_window_size]
        try:  # hava data-60s in this time domain d1 to d2
            type_B_arr = cal_attributes_B(data_array=seismic_data, sps=sps)
            type_A_arr = cal_attributes_A(data_array=seismic_data)
        except Exception as e:  # NOT hava any data-60s in this time domain d1 to d2
            type_B_arr = np.full(59, np.nan)
            type_A_arr = np.full(17, np.nan)
            print(f"NaN in {input_year}-{input_station}-{input_component}-{time}, \n"
                  f"Exception {e} \n")

        arr_RF, arr_BL = np.append(id, type_B_arr), np.append(id, type_A_arr)
        record_data(input_year, input_station, input_component, arr_RF, "B", julday, folder_path_txt)  # write data-60s by custom function
        record_data(input_year, input_station, input_component, arr_BL, "A", julday, folder_path_txt)  # write data-60s by custom function


def cal_loop(catchment_name, seismic_network, input_year, input_station, input_component, input_window_size,
             id1, id2, folder_path_txt, folder_path_npy):

    for julday in range(id1, id2):

        julday = str(julday).zfill(3)
        data_start = UTCDateTime(year=input_year, julday=julday)
        data_end = data_start + 24 * 3600

        st = load_seismic_signal(catchment_name, seismic_network, input_station, input_component, data_start, data_end)

        # dump the seismic data-60s
        npy_dir = f"{folder_path_npy}/{input_year}_{input_station}_{input_component}_{julday}"
        num_data = int(24 * 3600 * st[0].stats.sampling_rate)
        np.savez(npy_dir, sampling_rate=st[0].stats.sampling_rate, data=st[0].data[:num_data], window_size=input_window_size)

        # write the seismic features header
        record_data_header(input_year, input_station, input_component, julday, folder_path_txt)
        loop_time_step(st, input_year, input_station, input_component, input_window_size, julday, folder_path_txt)

def cal_loop_seg(catchment_name, seismic_network, input_year, input_station, input_component, input_window_size,
                 id1, id2, folder_path_txt, folder_path_npy):

    julday = str(id1).zfill(3)
    data_start = UTCDateTime(year=input_year, julday=julday)
    data_end = data_start + 24 * 3600 - 1 # only one day's data-60s available

    st = load_seismic_signal(catchment_name, seismic_network, input_station, input_component, data_start, data_end)

    # write the seismic features header
    record_data_header(input_year, input_station, input_component, julday, folder_path_txt)

    loop_time_step(st, input_year, input_station, input_component, input_window_size, julday, folder_path_txt)



def main(catchment_name, seismic_network, input_year, input_station, input_component, input_window_size, id):
    '''
    Set the input parametres for main function

    Parameters:
    - input_year (int or string): calculate year
    - input_station (str): calculate seismic station
    - input_component (str): component of calculate seismic station
    - input_window_size (int): window size of calculate

    Returns:
    - No returns
    '''

    job_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    print(f"Start Job {job_id}: {input_year}, {input_station}: ", datetime.now().strftime("%Y-%m-%dT%H:%M:%S") )

    # check the folder
    try:
        folder_path_txt, folder_path_npy, folder_path_net = check_folder(catchment_name, seismic_network, input_year, input_station, input_component)
    except FileExistsError as e:
        print(f"{seismic_network}, {input_year}, {input_station}, {input_component}, {input_window_size}, {id}, \n"
              f"Exception {e}: Directory already exists, ignoring.")
    except Exception as e:
        print(f"{seismic_network, input_year, input_station, input_component, input_window_size, id}, \n"
              f"Exception {e}")

    #map_start_julday = {2013:147, 2014:91,  2017_1:140, 2018-2019:145, 2019:145, 2020:152}
    #map_end_julday   = {2013:245, 2014:273, 2017_1:183, 2018-2019:250, 2019:250, 2020:250}
    #id1, id2 = map_start_julday.get(input_year),  map_end_julday.get(input_year)
    id1, id2 = id, id + 1
    if seismic_network in ["CC"]: # data-60s segment
        cal_loop_seg(catchment_name, seismic_network, input_year, input_station, input_component, input_window_size,
                     id1, id2, folder_path_txt, folder_path_npy)
    else:
        cal_loop(catchment_name, seismic_network, input_year, input_station, input_component, input_window_size,
                 id1, id2, folder_path_txt, folder_path_npy)


    print(f"End Job {job_id}: {input_year}, {input_station}: ", datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--catchment_name", type=str, default="Illgraben", help="check the year")
    parser.add_argument("--seismic_network", type=str, default="9S", help="check the year")
    parser.add_argument("--input_year", type=int, default=2020, help="check the year")
    parser.add_argument("--input_station", type=str, default="ILL12", help="check the input_station")
    parser.add_argument("--input_component", type=str, default="EHZ", help="check the input_component")
    parser.add_argument("--input_window_size", type=int, default=60, help="check the calculate window size")
    parser.add_argument("--id", type=int, default=60, help="check the julday id")

    args = parser.parse_args()
    main(args.catchment_name, args.seismic_network,
         args.input_year, args.input_station, args.input_component, args.input_window_size, args.id)
