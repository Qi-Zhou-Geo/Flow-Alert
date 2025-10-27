#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2025-07-28
# __author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission

import yaml
import numpy as np
from datetime import datetime, timedelta, timezone

# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path
current_dir = Path(__file__).resolve().parent
# using ".parent" on "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys
sys.path.append(str(project_root))
# </editor-fold>


def synthetic_input4model(sub_window_size, window_overlap, trained_model_name, seq_length,
                          data_start_time="2020-01-01T12:00:00"):
    '''
    Create synthetic data array for the trained ML model

    Args:
        sub_window_size: int, the window size
        window_overlap: float, the overlap of two adjacent windows
        trained_model_name: str, format by f"{model}_{feature_type}"
        seq_length: int, sequence length in time domain
        data_start_time: str, format by "%Y-%m-%dT%H:%M:%S", the first time of the output_feature

    Returns:
        output_feature: numpy 2D array, shape by (seq_length, 2 + num_feature)
    '''

    start = datetime.fromisoformat(data_start_time).replace(tzinfo=timezone.utc)
    interval_seconds = sub_window_size * (1 - window_overlap)

    t_str = [(start + timedelta(seconds=i * interval_seconds)).strftime("%Y-%m-%dT%H:%M:%S")for i in range(seq_length)]
    t_str = np.array(t_str)

    t_float = [(start + timedelta(seconds=i * interval_seconds)).timestamp() for i in range(seq_length)]
    t_float = np.array(t_float)

    config_path = f"{project_root}/config/config_inference.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    selected = config[f'feature_type_{trained_model_name[-1]}']
    num_selected = len(selected)
    feature_arr = np.ones((seq_length, num_selected))

    synthetic_feature = np.concatenate((t_str.reshape(-1, 1), t_float.reshape(-1, 1), feature_arr), axis=1)

    return synthetic_feature