#!/usr/bin/python
# -*- coding: UTF-8 -*-


#__modification time__ = 2024-05-27
#__author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission

import os
import shutil
import argparse

import numpy as np
import pandas as pd

# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path
current_dir = Path(__file__).resolve().parent
# using ".parent" on "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent
import sys
sys.path.append(str(project_root))
# </editor-fold>


# import the custom functions
from calculate_features.define_path import check_folder


def delete_folder(catchment_name, seismic_network, input_station, input_year, input_component):

    folder_path_txt, folder_path_npy, folder_path_net = \
        check_folder(catchment_name, seismic_network, input_year, input_station, input_component)

    try:
        shutil.rmtree(folder_path_npy)
    except Exception as e:
        print(e)


def merge_files(input_file_dir, input_files, output_file):

    with open(output_file, 'w') as outfile:
        header_written = False

        for filename in input_files:
            file_path = os.path.join(input_file_dir, filename)

            # Open each file for reading
            with open(file_path, 'r') as infile:
                # Read the header
                header = infile.readline()

                # Write the header only once
                if not header_written:
                    outfile.write(header)
                    header_written = True

                # Write the remaining lines (skip the header)
                for line in infile:
                    outfile.write(line)


def main(catchment_name, seismic_network, input_year, input_station, input_component, id1, id2):
    folder_path_txt, folder_path_npy, folder_path_net = check_folder(catchment_name, seismic_network, input_year, input_station, input_component)

    # Type A
    output_file = f"{folder_path_txt}/{input_year}_{input_station}_{input_component}_all_A.txt"
    input_files = [f"{input_year}_{input_station}_{input_component}_{str(i).zfill(3)}_A.txt" for i in range(id1, id2 + 1)]
    merge_files(folder_path_txt, input_files, output_file)

    # Type B
    output_file = f"{folder_path_txt}/{input_year}_{input_station}_{input_component}_all_B.txt"
    input_files = [f"{input_year}_{input_station}_{input_component}_{str(i).zfill(3)}_B.txt" for i in range(id1, id2 + 1)]
    merge_files(folder_path_txt, input_files, output_file)

    # Type B network
    if seismic_network in ["9S"] and input_station != "synthetic12":
        input_file_dir = folder_path_net
        output_file = f"{folder_path_net}/{input_year}_{input_component}_all_network.txt"
        input_files = [f"{input_year}_{input_component}_{str(i).zfill(3)}_net.txt" for i in range(id1, id2 + 1)]
        merge_files(input_file_dir, input_files, output_file)
    else:
        pass

    # remove the npy file to unload the space
    delete_folder(catchment_name, seismic_network, input_station, input_year, input_component)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--catchment_name", type=str, default="Illgraben", help="check the year")
    parser.add_argument("--seismic_network", type=str, default="9S", help="check the year")
    parser.add_argument("--input_year", type=int, default=2020, help="check the year")
    parser.add_argument("--input_station", type=str, default="ILL12", help="check the input_station")
    parser.add_argument("--input_component", type=str, default="EHZ", help="check the input_component")
    parser.add_argument("--id1", type=int, default=1, help="check the julday_id1")
    parser.add_argument("--id2", type=int, default=365, help="check the julday_id1")

    args = parser.parse_args()
    main(args.catchment_name, args.seismic_network,
         args.input_year, args.input_station, args.input_component, args.id1, args.id2)
