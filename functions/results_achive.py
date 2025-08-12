#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-02-23
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission

import os
import sys

import numpy as np
import pandas as pd

# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path
current_dir = Path(__file__).resolve().parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent
import sys
sys.path.append(str(project_root))
# </editor-fold>


# import CONFIG_dir as a global variable
from config.config_dir import CONFIG_dir


def achieve_predicted_results(time_stamps, obs_y_label, pre_y_label, pre_y_pro,
                              input_station, model_type, feature_type, input_component, training_or_testing):

    assert len(time_stamps) == len(obs_y_label), f"check the len(time_stamps) == len(obs_y_label)"
    assert len(time_stamps) == len(pre_y_label), f"check the len(time_stamps) == len(pre_y_label)"
    assert len(time_stamps) == len(pre_y_pro), f"check the len(time_stamps) == len(pre_y_pro)"


    array1 = np.array(time_stamps)
    array2 = np.array(obs_y_label)
    array3 = np.array(pre_y_label)
    array4 = np.array(pre_y_pro)

    df = pd.DataFrame({
        'time_window_start': array1,
        'obs_y_label': array2,
        'pre_y_label': array3,
        'pre_y_pro': array4
    })

    # double set the path
    output_path =  f"{CONFIG_dir['output_dir']}/trained_model/predicted_results/"
    os.makedirs(name=output_path, exist_ok=True)
    df.to_csv(f"{output_path}{input_station}_{model_type}_{feature_type}_{input_component}_{training_or_testing}_output.txt",
        index=False)

