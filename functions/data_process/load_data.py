#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-02-23
#__author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
#__find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission

import yaml

import pytz
from datetime import datetime

import random
import pandas as pd
import numpy as np

# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path
current_dir = Path(__file__).resolve().parent
# using ".parent" on "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys
sys.path.append(str(project_root))
# </editor-fold>

# import the custom functions
#from functions.data_process.archive_data_h5 import load_hdf5
from functions.data_process.random_select_feature import generate_random_selected_feature_id
from functions.seismic.seismic_data_processing import config_snesor_parameter


def clip_df_columns(df):

    for col_names in ["alpha"]:
        if col_names in df.columns:
            df[col_names] = df[col_names].clip(lower=1, upper=10)


    for col_names in ["ES_0", "ES_1", "ES_2", "ES_3", "ES_4"]:
        if col_names in df.columns:
            # clip -5.22 (~1e-7 m/s with 60s window) to -1.22 (~1e-3 m/s with 60s window)
            df[col_names] = df[col_names].clip(lower=-5.22, upper=-1.22)

    for col_names in ["env_max_to_duration"]:
        if col_names in df.columns:
            df[col_names] = df[col_names].clip(lower=0, upper=2e-5)

    for col_names in ["RMS", "IQR"]:
        if col_names in df.columns:
            # clip lower = 0 to 1e-4 = 100 * 1e-6 m/s
            df[col_names] = df[col_names].clip(lower=0, upper=1e-4)

    for col_names in ["MaxFFT"]:
        if col_names in df.columns:
            # clip lower = 0 to 1e-4 = 100 * 1e-6 m/s
            df[col_names] = df[col_names].clip(lower=0, upper=4e-9)

    for col_names in ["DistMaxMean", "DistMaxMedian"]:
        if col_names in df.columns:
            # clip lower = 0 to 1e-4 = 100 * 1e-6 m/s
            df[col_names] = df[col_names].clip(lower=0, upper=1e-9)

    return df

## for seismic feature
def load_all_features(catchment_name, seismic_network, input_year, input_station, input_component,
                      with_network, with_label,
                      normalize=False):

    # set dir path
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent


    # catchment mapping
    sac_path, feature_path, response_type, sensor_type = config_snesor_parameter(catchment_name, seismic_network)
    feature_dir = f"{feature_path}/{input_year}/{input_station}/{input_component}"
    network_feature_dir = f"{feature_path}/{input_year}/{input_component}_net"


    # BL sets
    df1 = pd.read_csv(f"{feature_dir}/{input_year}_{input_station}_{input_component}_all_A.txt",
                      header=0, low_memory=False,
                      # 14 is 'goodness', 16 is 'magnitude_range', 17 is 'alpha'
                      usecols=[4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 17])

    # waveform, spectrum, spectrogram sets
    df2 = pd.read_csv(f"{feature_dir}/{input_year}_{input_station}_{input_component}_all_B.txt",
                      header=0, low_memory=False,
                      usecols=np.arange(4, 63))

    assert df1.shape[0] == df2.shape[0], f"df1.shape={df1.shape[0]}, !=,  df2.shape={df2.shape[0]}"

    # network sets
    if with_network is True:  # need the network feature
        df3 = pd.read_csv(f"{network_feature_dir}/{input_year}_{input_component}_all_network.txt",
                          header=0, low_memory=False, usecols=np.arange(3, 13))
    else:  # do NOT need the network feature, and generate synthetic data-60s to keep the df4 structure
        df3 = pd.DataFrame(1, index=range(df1.shape[0]), columns=range(np.arange(3, 13).size))
        df3.columns = ['id_maxRMS', 'id_minRMS', 'ration_maxTOminRMS', 'ration_maxTOminIQR', 'mean_coherenceOfNet',
                       'max_coherenceOfNet', 'mean_lagTimeOfNet', 'std_lagTimeOfNet', 'mean_wdOfNet', 'std_wdOfNet']

    # manually label
    if with_label is True:  # with manually label
        print(with_label)
        df4 = pd.read_csv(f"{feature_dir}/{input_year}_{input_station}_{input_component}_all_A.txt",
                          header=0, low_memory=False)
        amp_array = np.array(df4.iloc[:, :2])
        amp_array = amp_array[:, [1, 0]]
        amp_array = load_waveform_pro(amp_array, seismic_network, input_year,
                                      input_station, input_component,
                                      with_label=with_label)

        df4 = pd.DataFrame(amp_array, columns=["time_stamps", "time_window_start", "label_0nonDF_1DF"])
    else:  # without manually label
        df4 = pd.read_csv(f"{feature_dir}/{input_year}_{input_station}_{input_component}_all_A.txt",
                          header=0, low_memory=False)
        df4 = df4.iloc[:, :2]
        df4 = df4.iloc[:, [1, 0]]
        df4["label_0nonDF_1DF"] = 0


    df = pd.concat([df1, df2, df3, df4], axis=1, ignore_index=True)
    columnsName = np.concatenate([df1.columns.values, df2.columns.values, df3.columns.values, df4.columns.values])
    df.columns = columnsName

    df = clip_df_columns(df)

    if normalize is True:
        # min max normalize the data-60s
        #from functions.data_process.min_max_normalize_transformer import min_max_normalize
        #df.iloc[:, :-3] = min_max_normalize(df.iloc[:, :-3], input_station, feature_type="C")

        with np.load(f"{project_root}/data/scaler/normalize_factor4C.npz", "r") as f:
            min_factor = f["min_factor"]
            max_factor = f["max_factor"]

        X = df.iloc[:, :-3].to_numpy().astype(float)
        scaled = (X - min_factor) / (max_factor - min_factor)
        df.iloc[:, :-3] = scaled

    return df

def select_features(catchment_name, seismic_network, input_year, input_station, input_component, feature_type, with_label,
                    descending=False, repeat=1, normalize=False):
    '''

    Args:
        seismic_network: str, input station
        input_year: str, year of data-60s, e.g., "2014"
        input_station: str, station of data-60s, e.g., "ILL12"
        input_component: str, component of data-60s, e.g., "EHZ"
        feature_type: str, type of feature e.g., A, B, C, D, and E
        with_label: bool, with manually labled feature or not,
        descending: bool, for testing the num of input feature influences,
                    rank the feature importance by descending (the first one is most important),
        repeat: int, for random selecting the feature

    Returns:
        input_features_name: 1D numpy array, List[str]
        df: data-60s frame, column by [time_stamp_float, x, y]
    '''


    if feature_type == "A":  # BL sets
        selected_column = np.arange(0, 11).tolist()
        selected_column.extend([80, 81, 82])  # 80 is time float, 81 is time str, 82 is manually label
        with_network = False
    elif feature_type == "B":  # waveform, spectrum, spectrogram, and network sets
        selected_column = np.arange(11, 83).tolist()
        with_network = True
    elif feature_type == "C":  # A and B
        selected_column = np.arange(0, 83).tolist()
        with_network = True
    elif feature_type == "D":  # A and B (but without network features)
        selected_column = np.arange(0, 70).tolist()
        selected_column.extend([80, 81, 82])
        with_network = False
    elif feature_type in ["E", "F", "G", "H"]:  # selected from C

        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parent.parent
        config_path = (project_root / f"./config/config_inference.yaml").resolve()
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            selected_column = config[f"feature_type_{feature_type}"]
            print(selected_column)
        selected_column.extend([80, 81, 82])
        with_network = False
    elif feature_type.split("-")[0] == "X":  # selected based on feature IMP
        from config.config_dir import CONFIG_dir
        # "X-16-model" means that the Top 16 features are selected
        num_selected = int(feature_type.split("-")[1])
        feature_weight = np.load(f"{CONFIG_dir['output_dir2']}/feature_C_weight/"
                                 f"{str(input_station).replace('1', '0')}-{feature_type.split('-')[2]}-C.npy")

        if descending is True:
            # descending order, then select top "num_selected"
            feature_weight_selected = np.sort(feature_weight)[::-1][:num_selected]
        else:
            # ascending order, then select top "num_selected"
            feature_weight_selected = np.sort(feature_weight)[:num_selected]

        # find the original index in the 1-80 feature index
        selected_column = []
        for i in feature_weight_selected:
            selected_column.append(np.where(feature_weight == i)[0][0])
        selected_column.sort()
        selected_column.extend([80, 81, 82])
        with_network = True
    elif feature_type.split("-")[0] == "R":  # randomly selected
        # "K-16-model" means that randomly selected 16 features
        num_selected = int(feature_type.split("-")[1])
        selected_column = generate_random_selected_feature_id(repeat, num_selected, num_total_feature=80)
        print(f"{seismic_network, input_year, input_station, input_component, feature_type}\n"
              f"{selected_column}")

        selected_column.extend([80, 81, 82])
        with_network = True
    else:
        print(f"please check the {feature_type}")

    df = load_all_features(catchment_name, seismic_network, input_year, input_station, input_component,
                           with_network, with_label, normalize=normalize)
    df = df.iloc[:, selected_column]
    input_features_name = df.columns[:-3]

    time_stamp_float, time_stamp_str = df.iloc[:, -3], df.iloc[:, -2]
    x, y = df.iloc[:, :-3], df.iloc[:, -1]

    assert not np.any(np.isinf(x)), "Feature array contains infinite values (inf) in func 'select_features'."
    assert not np.any(np.isnan(x)), "Feature array contains NaN values in func 'select_features'."

    df = pd.concat([time_stamp_float, x, y], axis=1, ignore_index=True)
    df = np.array(df)

    return input_features_name, df

    ## for waveform

def load_waveform_pro(amp_array, seismic_network, input_year,
                      input_station, input_component,
                      with_label=False, buffer=0):
    '''
    Receive numpy array and return the same array with debris flow probability/label

    Args:
        amp_array:
        seismic_network:
        input_year:
        input_station:
        input_component:
        with_label:
        buffer: unit is per data-60s

    Returns:


    '''

    # add one column as debris flow probability with default value 0
    temp_array = np.concatenate((amp_array, np.full(amp_array.shape[0], 0).reshape(-1, 1)), axis=1)  # axis=1 as column

    # if noone manually label the data-60s, set
    if with_label is False:
        return temp_array
    else:
        pass

    time_stamps = temp_array[:, 0]

    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent
    # df = pd.read_csv(f"{project_root}/data/manually_labeled_DF/{seismic_network}-{input_year}-DF.txt", header=0)
    df = pd.read_csv(f"{project_root}/data/event_catalog/{seismic_network}-{input_year}-DF.txt", header=0)
    # select the manually labeled event time
    df = df[(df.iloc[:, 4] == seismic_network) &
            (df.iloc[:, 5] == input_station) &
            (df.iloc[:, 6] == input_component)]

    for step in np.arange(len(df)):
        datetime_start = df.iloc[step, 2]
        timestamp_start = datetime.strptime(datetime_start, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=pytz.UTC).timestamp()

        datetime_end = df.iloc[step, 3]
        timestamp_end = datetime.strptime(datetime_end, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=pytz.UTC).timestamp()

        diff = np.abs(time_stamps - timestamp_start)
        id1 = np.where(diff == np.min(diff))[0][0] - buffer

        diff = np.abs(time_stamps - timestamp_end)
        id2 = np.where(diff == np.min(diff))[0][0] + buffer

        # use min-max to get the debris flow probability
        # amp_1_45 = temp_array[id1:id2, 1]
        # pro = (amp_1_45 - np.min(amp_1_45)) / (np.max(amp_1_45) - np.min(amp_1_45))
        pro = np.full((id2 - id1), 1)  # assign the debris flow as 1
        temp_array[id1:id2, -1] = pro

    print(f"Count label for: {seismic_network, input_year, input_station},"
          f"DF: {np.where(temp_array[:, -1] == 1)[0].shape},"
          f"Non-DF: {np.where(temp_array[:, -1] == 0)[0].shape},"
          f"All labels: {temp_array.shape[0]}")

    return temp_array