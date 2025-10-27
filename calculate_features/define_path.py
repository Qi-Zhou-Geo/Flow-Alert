#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-11-02
#__author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission

import os
import yaml


# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path
current_dir = Path(__file__).resolve().parent
# using ".parent" on "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent
import sys
sys.path.append(str(project_root))
# </editor-fold>

# import the custom functions
from functions.seismic.seismic_data_processing import config_snesor_parameter

def check_folder(catchment_name, seismic_network, input_year, input_station, input_component):

    # catchment mapping
    sac_path, feature_path, response_type, sensor_type = config_snesor_parameter(catchment_name, seismic_network)

    # create the folder
    folder_path_txt = f"{feature_path}/{input_year}/{input_station}/{input_component}"
    os.makedirs(folder_path_txt, exist_ok=True)

    folder_path_npy = f"{feature_path}/{input_year}/{input_station}/{input_component}/npy"
    os.makedirs(folder_path_npy, exist_ok=True)

    folder_path_net = f"{feature_path}/{input_year}/{input_component}_net"
    os.makedirs(folder_path_net, exist_ok=True)

    return folder_path_txt, folder_path_npy, folder_path_net
