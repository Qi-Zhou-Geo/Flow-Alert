#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-12-26
#__author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
#__find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission

import numpy as np
from datetime import datetime
from sklearn.metrics import confusion_matrix

from filelock import FileLock


def dump_evaluate_matrix(be_saved_array, note=None, output_dir=None, output_name=None, dump=True):
    '''
    save the evaluate matrix as txt file

    Args:
        be_saved_array: 2D numpy array,
        note: str,
        output_dir: str,
        output_name: str,

    Returns:

    '''

    epsilon = 1e-8  # avoid zero division
    try:
        tn, fp, fn, tp = confusion_matrix(be_saved_array[:, 2], be_saved_array[:, 4]).ravel()
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1 = tp / (tp + 0.5 * fp + 0.5 * fn + epsilon)
        f2 = tp / (tp + fp + 0.5 * fn + epsilon)
    except Exception as e:
        tn, fp, fn, tp = [-1] * 4
        precision, recall, f1, f2 = [-1] * 4
        print(f"error, {e}")

    # elevate matix
    evaluate_matrix = [tn, fp, fn, tp,
                       float(f"{precision:.4f}"), float(f"{recall:.4f}"),
                       float(f"{f1:.4f}"), float(f"{f2:.4f}")]

    if dump is True:
        lock_path = f"{output_dir}/{output_name}.lock"
        # lock the file to avoid the information lost when multiple process
        with FileLock(lock_path):
            temp = f"{note}, {tn}, {fp}, {fn}, {tp}, {precision :.4f}, {recall :.4f}, {f1 :.4f}, {f2 :.4f} \n"
            with open(f"{output_dir}/{output_name}.txt", "a") as file:
                file.write(temp)

    return evaluate_matrix


def dump_model_prediction(be_saved_array, training_or_testing, output_dir, output_format):
    '''
    save the ML model predicted results as txt file

    Args:
        be_saved_array: 2D numpy array,
        training_or_testing: str,
        output_dir: str,
        output_format: str,

    Returns:
        no returns
    '''

    # make sure the column [:, 3] is predicted probability
    be_saved_array[:, 3] = np.round(be_saved_array[:, 3], 3)

    time_stamps_float = be_saved_array[:, 0]
    time_stamps_string = [datetime.utcfromtimestamp(ts).strftime("%Y-%m-%dT%H:%M:%S") for ts in time_stamps_float]
    time_stamps_string = np.array(time_stamps_string).reshape(-1, 1)

    temp = np.hstack((time_stamps_string, be_saved_array))

    np.savetxt(f"{output_dir}/{output_format}-{training_or_testing}-output-{temp[0, 0][:10]}.txt",
               temp, delimiter=',', fmt='%s', comments='',
               header="time_window_start,time_stamps,obs_y_pro,obs_y_label,pre_y_pro,pre_y_label")


def dump_as_row(output_dir, output_name, variable_str, *args):
    '''
    dump the variables to local

    Args:
        output_dir:
        output_name:
        variable_str: pass one as str
        *args: pass any str or float, then conect by ","

    Returns:

    '''

    lock_path = f"{output_dir}/{output_name}.lock"
    # lock the file to avoid the information lost when multiple process
    with FileLock(lock_path):
        with open(f"{output_dir}/{output_name}.txt", "a") as f:

            record = f"{variable_str}"
            if args:
                # append additional arguments
                record += ", " + ", ".join(str(arg) for arg in args)

            f.write(record + "\n")  # Write to file with newline
