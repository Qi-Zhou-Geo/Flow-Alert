#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-02-23
#__author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
#__find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission

import os
import argparse

import numpy as np
import pandas as pd
from datetime import datetime

# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path
current_dir = Path(__file__).resolve().parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys
sys.path.append(str(project_root))
# </editor-fold>

# import the custom functions
from functions.warning_strategy.strategy import manually_warning


def main(model_type, feature_type, input_component,
         class_weight=0.9, ratio=100000, buffer=120):

    pro_filter = 0
    seismic_network = "9S"
    input_station_list = ["ILL08", "ILL02", "ILL03"]

    for idx1, warning_threshold in enumerate(np.arange(0.1, 1.1, 0.1)):
        for idx2, attention_window_size in enumerate(np.arange(1, 21, 1)):

            warning_threshold = np.round(warning_threshold, 1)
            attention_window_size = np.round(attention_window_size, 0)

            warning_output = manually_warning(pro_filter, warning_threshold, attention_window_size,
                                              input_station_list, model_type, feature_type, input_component,
                                              class_weight, ratio,
                                              seismic_network, buffer)
            header = f"warning_threshold, attention_window_size, num_station, " \
                     f"model_type, feature_type, input_component, class_weight, ratio, " \
                     f"num_false_warning, num_missed_warning, total_increases_warning, mean_increases_warning," \
                     f"increased_warning_E1, increased_warning_E2, increased_warning_E3, increased_warning_E4, " \
                     f"increased_warning_E5, increased_warning_E6, increased_warning_E7a, increased_warning_E7b, " \
                     f"increased_warning_E8, increased_warning_E9, increased_warning_E10, increased_warning_E11"

            record = f"{warning_threshold}, {attention_window_size}, 3, " \
                     f"{model_type}, {feature_type}, {input_component}, {class_weight}, {ratio}, " \
                     f"{warning_output}"

            file_name = f"/home/qizhou/3paper/" \
                        f"3Diversity-of-Debris-Flow-Footprints/output2/network_warning/" \
                        f"{model_type}-{feature_type}-{input_component}-warning.txt"

            with open(file_name, "a") as f:
                f.write(header + "\n")
                f.write(record + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='input parameters')

    parser.add_argument("--model_type", default="Random_Forest", type=str, help="model type")
    parser.add_argument("--feature_type", default="C", type=str, help="feature type")
    parser.add_argument("--input_component", default="EHZ", type=str, help="seismic input_component")

    args = parser.parse_args()

    main(args.model_type, args.feature_type, args.input_component)

