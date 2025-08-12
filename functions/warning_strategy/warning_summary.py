#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-02-23
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission

import os
import sys
import argparse

import pandas as pd
import numpy as np
from obspy.core import UTCDateTime # default is UTC+0 time zone
from datetime import datetime


# Get the absolute path of the parent directory
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# internal functions
from config.config_dir import CONFIG_dir


def warning_summary(pro_threshold, warning_threshold, attention_window_size,
                    model_type, feature_type, input_component):
    # t1 minutess before the ILL18 start time, t2 minutes after the CD29 will not as false warning
    t1, t2 = 60, 180

    df1 = pd.read_csv(f"{project_root}/plotting/network_warning/output/{model_type}_{feature_type}_{input_component}_warning_"
                      f"{pro_threshold}_{warning_threshold}_{attention_window_size}.txt", header=0)
    date = np.array(df1.iloc[:, 0])

    df2 = pd.read_csv(f"{project_root}/plotting/network_warning/2020_testing_events.txt", header=None)
    df3 = pd.read_csv(f"{project_root}/plotting/network_warning/2020_CD29time.txt", header=None)

    temp = []
    false_warning = np.array(df1.iloc[:, -2])
    df1['increased_warning_time'] = df1['increased_warning_time'].replace(to_replace=['noise', 'false_warning'], value=0)

    for step in range(len(df2)):
        id1 = np.where(date == df2.iloc[step, 0])[0][0]
        id2 = np.where(date == df2.iloc[step, -1])[0][0]
        id3 = np.where(date == df3.iloc[step, 0][:-3]+":00")[0][0]

        # remove the false warning
        # 60 mins before the ILL18 start time, 180 minutes after the CD29
        false_warning[id1 - t1 : id3 + t2] = np.where(false_warning[id1 - t1 : id3 + t2] == 'false_warning',
                                                        'fake_warning', # if the donditation is Ture
                                                        false_warning[id1 - t1 : id3 + t2])# if the donditation is False

        # make sure only use the warning time before id3 (cd29 time stamps)
        # and the warning may start before the manually labeled start time
        increased_warning_time = np.array(df1.iloc[id1-20:id3+60, -2], dtype= float)
        if np.sum(increased_warning_time) > 0:
            temp.append(np.max(increased_warning_time)) # do not warry, the increased_warning_time decreases
        else:
            temp.append(0)

    false_warning_times = np.where(false_warning == 'false_warning')[0].size


    record = [model_type, feature_type, input_component,
              pro_threshold, warning_threshold, attention_window_size, "false_warning_times", false_warning_times,
              "increased_warning", np.sum(temp), np.max(temp), np.min(temp), len(temp) - np.count_nonzero(temp)]
    record.extend(list(temp))

    f = open(f"{project_root}/plotting/network_warning/warning_summary_{model_type}_{feature_type}_{input_component}.txt", 'a')
    f.write(str(record) + "\n")
    f.close()


    # replace the df1
    df1.iloc[:, -2] = false_warning
    df1.to_csv(f"{project_root}/plotting/network_warning/output/{model_type}_{feature_type}_{input_component}_warning_"
               f"{pro_threshold}_{warning_threshold}_{attention_window_size}.txt", sep=',', index=False, mode='w')

    return record


def dual_testing_warning_summary(pro_threshold, warning_threshold, attention_window_size,
                                 model_type, feature_type, input_component, seismic_network, input_data_year):
    # t1 minutess before the ILL18 start time, t2 minutes after the CD29 will not as false warning
    t1, t2 = 60, 180

    df1 = pd.read_csv(f"{CONFIG_dir['output_dir']}/dual_test_{seismic_network}_warning/{model_type}_{feature_type}_{input_component}_warning_"
                      f"{pro_threshold}_{warning_threshold}_{attention_window_size}.txt", header=0)
    date = np.array(df1.iloc[:, 0])



    df3 = pd.read_csv(f"{CONFIG_dir['project_root']}/data_input/warning_timestamp_benchmark/{input_data_year}_CD29time.txt", header=None)

    delta = []
    for step in range(len(df3)):
        id1 = np.where(date == df3.iloc[step, 0])[0][0] - 120
        id2 = np.where(date == df3.iloc[step, 0])[0][0]
        increased_warning_time = np.array(df1.iloc[id1:id2, -2], dtype= float)
        increased_warning_time = np.max(increased_warning_time)
        delta.append(increased_warning_time)

    false_warning_times = np.where(df1.iloc[:, -3] == 'fake_warning')[0].size


    record = [model_type, feature_type, input_component,
              pro_threshold, warning_threshold, attention_window_size, "false_warning_times", false_warning_times]
    record.extend(list(delta))

    f = open(f"{CONFIG_dir['output_dir']}/dual_test_{seismic_network}_warning/dual_testing_warning_summary.txt", 'a')
    f.write(str(record) + "\n")
    f.close()

    return record

