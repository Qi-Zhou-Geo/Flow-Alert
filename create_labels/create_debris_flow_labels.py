#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-02-23
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission

import os

import argparse

import pandas as pd
import numpy as np
from obspy.core import UTCDateTime # default is UTC+0 time zone



# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path
current_dir = Path(__file__).resolve().parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent
import sys
sys.path.append(str(project_root))
# </editor-fold>

# import the self-define functions
from config.config_dir import CONFIG_dir, path_mapping


def create_label(seismic_network, input_year, input_station, input_component, usecols):

    #folder_path_in = f"{project_root}/data_input/seismic_feature/{path_mapping(seismic_network)}/" \
                     #f"{input_year}/{input_station}/{input_component}"
    folder_path_in = f"{project_root}/data_output/seismic_feature"
    folder_path_out = f"{project_root}/data_output/event_label"

    df0 = pd.read_csv(f"{folder_path_in}/{input_year}_{input_station}_{input_component}_all_A.txt", header=0)

    folder_path = f"{project_root}/data_input/manually_labeled_DF"
    df1 = pd.read_csv(f"{folder_path}/{input_year}_DF_{seismic_network}.txt", header=0, usecols=usecols)

    df_label = df0.iloc[:, :2]
    df_label["label_0nonDF_1DF"] = 0 # non-debris flow event with label 0
    date = np.array(df_label.iloc[:, 0])

    for step in range(len(df1)):
        id1 = df1.iloc[step, 0]
        id2 = df1.iloc[step, 1]

        id1 = np.where(date == id1)[0][0]
        id2 = np.where(date == id2)[0][0]

        df_label.iloc[id1:id2, -1] = np.full(id2-id1, 1) # debris flow event with label 1

    os.makedirs(folder_path_out, exist_ok=True)
    df_label.to_csv(f"{folder_path_out}/{input_year}_{input_station}_{input_component}_observed_label.txt",index=False)

    print(f"{input_year}_{input_station}_{input_component}, num_label1={sum(df_label.iloc[:, -1])}, done")

def main(seismic_network, input_year, input_station, input_component, usecols):

    create_label(seismic_network, input_year, input_station, input_component, usecols)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='input parameters')

    parser.add_argument("--seismic_network", type=str, default="9S", help="combined_params")
    parser.add_argument("--input_year", default=2020, type=int)
    parser.add_argument("--input_station", default="C", type=str)
    parser.add_argument("--input_component", default="EHZ", type=str)
    parser.add_argument("--usecols", nargs='+', type=int, help="list of stations")

    args = parser.parse_args()

    main(args.seismic_network, args.input_year, args.input_station, args.input_component, args.usecols)

