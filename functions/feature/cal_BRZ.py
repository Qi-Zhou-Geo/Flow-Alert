#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2026-05-03
# __author__ = Qi Zhou and Sibashish Dash, GFZ Helmholtz Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission

import os
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from obspy import UTCDateTime, read

# region ### add the sys.path to search for custom modules ###
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
current_dir = current_file.parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent

sys.path.append(str(project_root))
# endregion

# import the custom functions
from func.seismic.seismic_data_processing import load_seismic_signal
from func.feature.Type_A_features import calBL_feature
from func.feature.Type_B_features import calculate_all_attributes
from func.feature.feature_name import load_feature_name


def cal_attributes_A(data_array, scaling=1e9, ruler=100): # the main function is from Qi
    data_array_nm = data_array * scaling # converty m/s to nm/s
    # the physical velocity of lower bodunday = 1e-9 m/s and upper boundary as 1e-4
    data_array_nm = np.clip(data_array_nm, a_min=1, a_max=1e5)
    feature_array = calBL_feature(data_array_nm, ruler)

    return feature_array


def cal_attributes_B(data_array, sps): # the main function is from Clement
    # sps: sampling frequency; flag=0: one component seismic signal;
    features = calculate_all_attributes(Data=data_array, sps=sps, flag=0)[0] # feature 1 to 60
    feature_array = features[1:]# leave features[0]=event_duration

    return feature_array # 59 features


def loop_24h_data(st, sub_window_size):
    
    # columns array to store the seismic data-60s for network features
    sps = int(st[0].stats.sampling_rate)

    input_year = st[0].stats.starttime.year
    julday = st[0].stats.starttime.julday

    input_station = st[0].stats.station
    input_component = st[0].stats.channel

    total_seconds = float(st[0].stats.endtime - st[0].stats.starttime)
    num_time_step = int(total_seconds / sub_window_size) # e.g., 1440 = (24h * 60 minutes) / 1 minute
    
    d0 = UTCDateTime(year=input_year, julday=julday)  # the start day, e.g.2014-07-12T00:00:00

    # prepare container
    feature_Name_A, feature_Name_B = load_feature_name()
    arr_A = np.empty(shape=(num_time_step, len(feature_Name_A)), dtype="object")
    arr_B = np.empty(shape=(num_time_step, len(feature_Name_B)), dtype="object")

    for step in tqdm(range(num_time_step), 
                     desc=f"Loop st: {d0}", 
                     total=num_time_step, 
                     file=sys.stdout):
        
        d1 = d0 + (step) * sub_window_size      # from minute 0, 1, 2, 3, 4
        d2 = d0 + (step + 1) * sub_window_size  # from minute 1, 2, 3, 4, 5
        d1_str = UTCDateTime(d1).strftime("%Y-%m-%dT%H:%M:%S")

        # metadata for calculated seismic features
        meta_data = np.array([d1_str, d1.timestamp, input_station, input_component])
        num_meta = len(meta_data)
        
        tr = st.copy()
        try:
            tr.trim(starttime=d1, endtime=d2, nearest_sample=False)
            seismic_data = tr[0].data[:sps * sub_window_size]
            
            type_A_arr = cal_attributes_A(data_array=seismic_data)
            type_B_arr = cal_attributes_B(data_array=seismic_data, sps=sps)
        except Exception as e:
            type_A_arr = np.full(shape=len(feature_Name_A) - num_meta, fill_value=np.nan)
            type_B_arr = np.full(shape=len(feature_Name_B)- num_meta, fill_value=np.nan)
            print(f"Warning!\n{UTCDateTime.now().isoformat()}\n"
                  f"Error for time {d1} to {d2}:{e}\n"
                  f"Fill gap as: np.nan\n")

        arr_A_temp = np.append(meta_data, type_A_arr)
        arr_B_temp = np.append(meta_data, type_B_arr)
        
        arr_A[step, :] = arr_A_temp
        arr_B[step, :] = arr_B_temp
    
    # warp to df
    df_A = pd.DataFrame(data=arr_A, columns=feature_Name_A)
    df_B = pd.DataFrame(data=arr_B, columns=feature_Name_B)
    
    idx_component = np.where(np.array(feature_Name_B) == "component")[0][0] + 1
    df = pd.concat((df_A, df_B.iloc[:, idx_component:]), axis=1) # concatenate along columns
    
    return df


def main(
    year,
    julday,
    
    seismic_network,
    station,
    component,
    
    f_min,
    f_max,

    sub_window_size):

    data_start = UTCDateTime(year=year, julday=julday).strftime("%Y-%m-%dT%H:%M:%S")
    data_end = UTCDateTime(year=year, julday=julday + 1).strftime("%Y-%m-%dT%H:%M:%S")
    
    # (1) load 24 hours data
    st = load_seismic_signal(
        seismic_network,
        station,
        component,
        data_start,
        data_end,
        f_min,
        f_max,
        remove_sensor_response=True,
        raw_data=False,
    )

    # (2) run the cal feature model
    df = loop_24h_data(st, sub_window_size)
    
    # (3) save to local
    output_dir = Path(project_root) / "data" / seismic_network / station / component
    os.makedirs(output_dir, exist_ok=True)
    output_name = f"{julday:03d}_{seismic_network}_{station}_{component}_f_min={f_min}_f_max={f_max}"
    df.to_csv(f"{output_dir}/{output_name}.txt", index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='input parameters')

    parser.add_argument("--seismic_network", type=str, default="BRZ")
    parser.add_argument("--station", type=str, default="02")
    parser.add_argument("--component", type=str, default="BHZ")

    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--julday", type=int, default=166) # UTC 2023-06-15T21:38:00 (julday=166) >> main ls failure
    parser.add_argument("--sub_window_size", type=int, default=2) # Unit by second

    parser.add_argument("--f_min", type=int, default=1)
    parser.add_argument("--f_max", type=int, default=45)

    args = parser.parse_args()


    main(
        year=args.year,
        julday=args.julday,
        seismic_network=args.seismic_network,
        station=args.station,
        component=args.component,
        f_min=args.f_min,
        f_max=args.f_max,
        sub_window_size=args.sub_window_size,
    )
