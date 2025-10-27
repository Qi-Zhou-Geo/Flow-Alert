#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2024-02-23
# __author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this functions without the author's permission

import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix, f1_score


# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path

current_dir = Path(__file__).resolve().parent
# using ".parent" on "pathlib.Path" object_typeect moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys

sys.path.append(str(project_root))
# </editor-fold>

# import the custom functions

def buffer_model_prediction(obs_y_label, buffer_l, buffer_r):
    '''
    Buffer model prediction

    Args:
        obs_y_label: 1D numpy array or pandas dataframe,
        buffer_l: int or float, unit by sub-window length
        buffer_r: int or float, unit by sub-window length

    Returns
        tolerance_mask: 1D numpy array of boolean value,
                        True -> maksed time step,
                        False -> not maksed

        FP inside tolerance_mask → TN, and FN inside buffer → TP
    '''
    num_obs = len(obs_y_label)

    tolerance_mask = np.zeros(num_obs, dtype=bool)
    event_indices = np.where(obs_y_label == 1)[0]

    for idx in event_indices:
        start = max(0, idx - buffer_l)
        end = min(num_obs, idx + buffer_r + 1)
        tolerance_mask[start:end] = True

    return tolerance_mask

def cal_buffered_cm(obs_y_label, pre_y_label, buffer_l=5, buffer_r=60, log_results=True):
    '''
    Calculate buffered confusion matrix

    Args:
        obs_y_label: 1D numpy array or pandas dataframe, time-series labele
        pre_y_label: 1D numpy array or pandas dataframe, time-series labele
        tolerance_mask: 1D numpy array, time-series boolean value
        buffer_l: int or float, buffer length in left side of an event, unit by sub-window length
        buffer_r: int or float, buffer length in right side of an event, unit by sub-window length

    Returns:
        cm_buffered,
        f1_buffered
    '''

    tolerance_mask = buffer_model_prediction(obs_y_label, buffer_l=buffer_l, buffer_r=buffer_r)


    # Un-buffered results
    cm = confusion_matrix(y_true=obs_y_label, y_pred=pre_y_label)
    f1_unbuffered = f1_score(y_true=obs_y_label, y_pred=pre_y_label, average='binary')

    if log_results is True:
        print(f"Un-buffered CM:\n"
              f"{cm}\n"
              f"F1={f1_unbuffered:.3f}")

    # start the buffer process
    obs = obs_y_label.copy()
    pre = pre_y_label.copy()

    # FP inside buffer → TN
    mask_fp_buffered = (pre == 1) & (obs == 0) & tolerance_mask
    pre[mask_fp_buffered] = 0

    # FN inside buffer → TP
    # mask_fn_buffered = (pre == 0) & (obs == 1) & tolerance_mask
    # pre[mask_fn_buffered] = 1

    cm_buffered = confusion_matrix(y_true=obs, y_pred=pre)
    f1_buffered = f1_score(y_true=obs, y_pred=pre, average='binary')
    if log_results is True:
        print(f"Buffered CM:\n"
              f"{cm_buffered}\n"
              f"F1={f1_buffered:.3f}")

    # FP inside buffer → TN
    FP_buffered = ((pre_y_label == 1) &
                   (obs_y_label == 0) &
                   (~tolerance_mask)).astype(int) * 1  # * 1 for better visualization
    # FN inside buffer → TP
    FN_buffered = ((pre_y_label == 0) &
                   (obs_y_label == 1) &
                   (~tolerance_mask)).astype(int) * -1


    return cm_buffered, f1_buffered
