#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2025-08-12
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission

import os
import argparse

import yaml

import shutil
from datetime import datetime


def rename_model(ML_name, station, feature_type, batch_size, seq_length, num_repeat):

    # <editor-fold desc="add the sys.path to search for custom modules">
    from pathlib import Path
    current_dir = Path(__file__).resolve().parent
    # using ".parent" on "pathlib.Path" object moves one level up the directory hierarchy
    project_root = current_dir.parent.parent

    import sys
    sys.path.append(str(project_root))
    # </editor-fold>

    # laod the pre-defined model format
    config_path = f"{project_root}/config/config_inference.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # "Model=[ML_name]_STA=[station]_Feature=[feature_type]_repeat=[num_repeat].pt"
    params = config[f"{ML_name}_{feature_type}"]

    model_name_format = params[f"ref_model_name"]
    feature_size = params[f"feature_size"]

    model_name_format = model_name_format.replace("[ML_name]", str(ML_name))
    model_name_format = model_name_format.replace("[station]", str(station))
    model_name_format = model_name_format.replace("[feature_type]", str(feature_type))
    model_name_format = model_name_format.replace("[num_repeat]", str(num_repeat))

    return feature_size, model_name_format


def copy_model(path_in, path_out, current_pt_name, num_repeat):

    temp = current_pt_name.split("-")
    ML_name, station, feature_type = temp[10], temp[3], temp[5]
    station = station.replace("IGB", "ILL")
    batch_size, seq_length = temp[12][1:], temp[13][1:]

    for n in range(1, num_repeat+1):

        pt_name = current_pt_name.replace("repeat-1", f"repeat-{n}")
        src = f"{path_in}/{pt_name}.pt"

        feature_size, model_name_format = rename_model(ML_name, station, feature_type, batch_size, seq_length, num_repeat=n)
        dst = f"{path_out}/{model_name_format}" # model_name_format with extension ".pt"

        shutil.copy(src, dst)

        if n == 1:
            mode = "w"
        else:
            mode = "a"

        time_now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        temp = f"Time: {time_now} \n" \
               f"copy: {src}\n" \
               f"to: {dst}\n \n"

        with open(f"{path_out}/ReadMe4Model_Details.txt", mode) as file:
            file.write(temp)


def main(path_in, path_out, current_pt_name, num_repeat):
    os.makedirs(path_out, exist_ok=True)
    copy_model(path_in, path_out, current_pt_name, num_repeat)

if __name__ == "__main__":
    # sinfo -n node[501-514] -N --Format="Nodelist,CPUsState,AllocMem,Memory,GresUsed,Gres"
    parser = argparse.ArgumentParser(description='input parameters')

    parser.add_argument("--path_in", type=str)
    parser.add_argument("--path_out", type=str)

    parser.add_argument("--current_pt_name", type=str)

    parser.add_argument("--num_repeat", type=int)

    args = parser.parse_args()

    main(args.path_in, args.path_out,
         args.current_pt_name, args.num_repeat)
