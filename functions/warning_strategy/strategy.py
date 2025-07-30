#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-02-23
#__author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
#__find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission

import os
import argparse

import pandas as pd
import numpy as np
from datetime import datetime

# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path
current_dir = Path(__file__).resolve().parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys
sys.path.append(str(project_root))
# </editor-fold>



def selected_data(station, model, feature,
                  data_compression = True,
                  input_component="EHZ", num_repeate=5, class_weight=0.9, ratio=100000,
                  time1="2020-05-29T01:00:00", time2="2020-09-06T23:00:00"):
    '''

    Args:
        station:
        model:
        feature:
        data_compression:
        input_component:
        num_repeate:
        class_weight:
        ratio:
        time1:
        time2:

    Returns:

    '''

    df_temp = None
    file_path = "/storage/vast-gfz-hpc-01/home/qizhou/3paper/3Diversity-of-Debris-Flow-Footprints/output-reviserved-paper"#f"{CONFIG_dir['output_dir2']}"
    for repeate in np.arange(1, num_repeate+1):
        name = f"9S-2017-{station}-{input_component}-training-True-repeate-{repeate}-" \
               f"{model}-{feature}-DFweight-{class_weight}-" \
               f"ratio-{ratio}-testing-output-2020-05-29.txt"

        temp = pd.read_csv(f"{file_path}/{model}/{name}", header=0)
        time_date = np.array(temp.iloc[:, 0])

        id1 = np.where(time_date == time1)[0][0]
        id2 = np.where(time_date == time2)[0][0] + 1

        temp = temp.iloc[id1:id2, :]

        if repeate == 1:
            df_temp = np.array(temp.iloc[:, [0, 1, 4]])
        else:
            df_temp = np.hstack((df_temp, np.array(temp.iloc[:, 4]).reshape(-1, 1) ))

    output = None
    if data_compression is True:
        output = df_temp[:, :2]
        mean_pro = np.mean(df_temp[:, 2:], axis=1) # as row, use the mean pro from "num_repeate"
        output = np.hstack((output, mean_pro.reshape(-1, 1)))
    else:
        output = df_temp

    return output

def warning_controller(pro_arr, warning_threshold, pro_filter=0):
    '''
    Check the wanring status

    Args:
        pro_arr: numpy array, model predicted probability
        warning_threshold: float, threshold for issuing the warning
        pro_filter: float, whether only consider the predicted value bigger than "pro_filter"

    Returns:
        status: str
    '''

    global_mean_pro = np.sum(pro_arr[pro_arr > pro_filter]) / pro_arr.size

    if global_mean_pro > warning_threshold:
        status = "warning"
    else:
        status = "noise"

    return status


def check_warning(time_str, warning_status,
                  seismic_network, input_station, input_component,
                  buffer, print_false_warning=True):

    date = time_str
    status = warning_status
    input_year = date[0][:4]

    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent

    file_dir = f"{project_root}/config/manually_labeled_DF"
    event = pd.read_csv(f"{file_dir}/{seismic_network}-{input_year}-DF.txt", header=0)
    # select the manually labeled event time
    event = event[(event.iloc[:, 4] == seismic_network) &
                  (event.iloc[:, 5] == input_station) &
                  (event.iloc[:, 6] == input_component)]

    cd29_time = pd.read_csv(f"{file_dir}/{seismic_network}-{input_year}-CD29-time.txt", header=0)
    cd29_time = np.array(cd29_time).reshape(-1)

    assert event.shape[0] == cd29_time.shape[0], f"event.shape[0] != cd29_time.shape[0], " \
                                                 f"{event.shape[0], cd29_time.shape[0]}"

    increased_warning = [] # unit by seconds
    missed_warning = []
    for idx, (event_start, event_end) in enumerate(zip(event.iloc[:, 2], event.iloc[:, 3])):

        id1 = np.where(date == event_start)[0][0] - buffer
        id2 = np.where(date == event_end)[0][0] + 1 + buffer
        temp_status = status[id1:id2] # current status for event (event_start, event_end)

        if 1 in temp_status or 10 in temp_status:
            # with warning
            id3 = np.where((temp_status == 1) | (temp_status == 10))[0][0]
            status[id1:id2][temp_status == 1] = 10 # replace the status 1

            # calculate the increased warning
            t1 = datetime.strptime(date[id1+id3], "%Y-%m-%dT%H:%M:%S")
            t2 = datetime.strptime(cd29_time[idx], "%Y-%m-%dT%H:%M:%S")
            delta_t = (t2 - t1).total_seconds()
        else:
            # no warning
            delta_t = np.nan
            missed_warning.append(event_start)

        increased_warning.append(delta_t)

    status[status == 1] = -1 # false warning
    status[status == 10] = 1 # set ture warning status back to 1

    num_false_warning = np.where(status == -1)[0].size
    num_missed_warning = len(missed_warning)

    if np.all(np.isnan(increased_warning)) is True:
        # all event are not warned, increased_warning is full "np.nan"
        total_increases_warning = 0
        mean_increases_warning = 0
    else:
        total_increases_warning = np.nansum(np.array(increased_warning))
        mean_increases_warning = np.nanmean(np.array(increased_warning))

    if print_false_warning is True:
        for false_warning in date[np.where(status == -1)]:
            print(f"false_warning at {false_warning}")

    output = f"{num_false_warning}, {num_missed_warning}, " \
             f"{total_increases_warning}, {mean_increases_warning}," \
             f"{increased_warning}"

    return output


def manually_warning(pro_filter, warning_threshold, attention_window_size,
                     input_station_list, model_type, feature_type, input_component,
                     class_weight, ratio,
                     seismic_network="9S", buffer=120):

    # load the predicted DF probability
    predicted_pro = None
    for idx, station in enumerate(input_station_list):
        pro = selected_data(station, model=model_type, feature=feature_type,
                            class_weight=class_weight, ratio=ratio)

        if idx == 0:
            predicted_pro = pro
        else:
            predicted_pro = np.hstack((predicted_pro, pro[:, 2].reshape(-1, 1)))

    warning_status_mapping = {"noise":0, "warning":1}
    warning_status = [] # 0 noise, 1 true warning, -1 false warning

    for step in range(attention_window_size, predicted_pro.shape[0]):

        pro_arr = predicted_pro[step-attention_window_size:step, 2:]
        status = warning_controller(pro_arr, warning_threshold, pro_filter)
        warning_status.append(warning_status_mapping.get(status))

    time_str = predicted_pro[attention_window_size:, 0]
    warning_status = np.array(warning_status, dtype=float)
    input_station = str(input_station_list[1].replace("0", "1"))

    warning_output = check_warning(time_str, warning_status, seismic_network, input_station, input_component, buffer)

    return warning_output


def quick_check_warning(predicted_pro, attention_window_size, warning_threshold, pro_filter):
    '''

    Args:
        predicted_pro: 2D numpy array, stacked model predicted probability
        attention_window_size: int or float

    Returns:

    '''
    warning_status_mapping = {"noise": 0, "warning": 1}
    warning_status = []

    for step in range(attention_window_size, predicted_pro.shape[0]):

        pro_arr = predicted_pro[step-attention_window_size:step, :]
        status = warning_controller(pro_arr, warning_threshold, pro_filter)
        warning_status.append(warning_status_mapping.get(status))

    output = np.array(warning_status)

    return output