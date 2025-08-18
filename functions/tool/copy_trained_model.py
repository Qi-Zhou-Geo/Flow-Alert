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
    # using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
    project_root = current_dir.parent.parent

    import sys
    sys.path.append(str(project_root))
    # </editor-fold>

    # laod the pre-defined model format
    config_path = f"{project_root}/config/config_inference.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # "Model=[ML_name]_STA=[station]_Feature=[feature_type]_b=[batch_size]_s=[seq_length]_repeat=[num_repeat].pt"
    params = config[f"{ML_name}_{feature_type}"]

    model_name_format = params[f"ref_model_name"]
    feature_size = params[f"feature_size"]

    model_name_format = model_name_format.replace("[ML_name]", str(ML_name))
    model_name_format = model_name_format.replace("[station]", str(station))
    model_name_format = model_name_format.replace("[feature_type]", str(feature_type))
    model_name_format = model_name_format.replace("[batch_size]", str(batch_size))
    model_name_format = model_name_format.replace("[seq_length]", str(seq_length))
    model_name_format = model_name_format.replace("[num_repeat]", str(num_repeat))

    return feature_size, model_name_format


def copy_model(path_in, path_out, ML_name, batch_size, seq_length, station, feature_type, num_repeat):


    for n in range(1, num_repeat+1):

        src = f"{path_in}/Illgraben-9S-2017-{station}-EHZ-{feature_type}-training-True-repeat-{n}-{ML_name}-{feature_type}-b{batch_size}-s{seq_length}.pt"

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

def main(path_in, path_out, ML_name, batch_size, seq_length, station, feature_type, num_repeat):

    copy_model(path_in, path_out, ML_name, batch_size, seq_length, station, feature_type, num_repeat)

if __name__ == "__main__":
    # sinfo -n node[501-514] -N --Format="Nodelist,CPUsState,AllocMem,Memory,GresUsed,Gres"
    parser = argparse.ArgumentParser(description='input parameters')

    parser.add_argument("--path_in", type=str)
    parser.add_argument("--path_out", type=str)

    parser.add_argument("--ML_name", type=str)

    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--seq_length", type=int)

    parser.add_argument("--station", type=str)
    parser.add_argument("--feature_type", type=str)

    parser.add_argument("--num_repeat", type=int)

    args = parser.parse_args()

    main(args.path_in, args.path_out,
         args.ML_name,

         args.batch_size, args.seq_length,

         args.station, args.feature_type,

         args.num_repeat)

#