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
from functions.issue_network_warning.warning_strategy import *

def asd(model_type, feature_type, input_component):

    print(model_type, feature_type, input_component)

    pro_threshold = 0
    input_station_list = ["ILL18", "ILL12", "ILL13"]

    for idx1, warning_threshold in enumerate(np.arange(0.1, 1.1, 0.1)):
        for idx2, attention_window_size in enumerate(np.arange(1, 21, 1)):

            warning_threshold = np.round(warning_threshold, 1)
            attention_window_size = np.round(attention_window_size, 0)

            warning(pro_threshold, warning_threshold, attention_window_size,
                    input_station_list, model_type, feature_type, input_component)

            record = warning_summary(pro_threshold, warning_threshold, attention_window_size,
                                     model_type, feature_type, input_component)

            print(f"Finish, {idx1}--{idx2}, {record} {pro_threshold, warning_threshold, attention_window_size, model_type, feature_type, input_component}",
                  datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

def main(model_type, feature_type, input_component):
    input_data_year = 2022
    seismic_network = "9S"

    df_write = pd.DataFrame()

    for idx1, warning_threshold in enumerate(np.arange(0.1, 1.1, 0.1)):
        for idx2, attention_window_size in enumerate(np.arange(1, 21, 1)):

            warning_threshold = np.round(warning_threshold, 1)
            attention_window_size = np.round(attention_window_size, 0)

            warning(0, warning_threshold, attention_window_size, ["ILL17", "ILL12", "ILL13"], model_type, feature_type, input_component, input_data_year)
            record = dual_testing_warning_summary(0, warning_threshold, attention_window_size, model_type, feature_type, input_component, seismic_network, input_data_year)
            print(record)
            df_write = df_write.append([record], ignore_index=True)

    df_write.to_csv(f"{CONFIG_dir['output_dir']}/dual_test_{seismic_network}_warning/dual_testing_warning_summary2.txt",
                    sep=',', header=False, index=False, mode='a')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='input parameters')

    parser.add_argument("--model_type", default="Random_Forest", type=str, help="model type")
    parser.add_argument("--feature_type", default="C", type=str, help="feature type")
    parser.add_argument("--input_component", default="EHZ", type=str, help="seismic input_component")

    args = parser.parse_args()

    main(args.model_type, args.feature_type, args.input_component)

