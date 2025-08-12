#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-02-23
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission

import os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from datetime import datetime

def summary_results(output_dir, input_station, model_type, feature_type, input_component, training_or_testing, num_feats):

    # <editor-fold desc="add the sys.path to search for custom modules">
    from pathlib import Path
    current_dir = Path(__file__).resolve().parent
    # using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
    project_root = current_dir.parent
    import sys
    sys.path.append(str(project_root))
    # </editor-fold>

    if input_station == "ILL08" or input_station == "ILL18":
        usecols = [2, 3]
    elif input_station == "ILL02" or input_station == "ILL12":
        usecols = [5, 6]
    elif input_station == "ILL03" or input_station == "ILL13":
        usecols = [7, 8]
    else:
        print(f"check the input station: {input_station}")

    if training_or_testing == "training":
        skiprows = 0
        nrows = 20
    elif training_or_testing == "testing":
        skiprows = 20
        nrows = 12
    else:
        print(f"please check the training_or_testing, {training_or_testing}")

    folder_path = f"{project_root}/data_input/manually_labeled_DF"
    df1 = pd.read_csv(f"{folder_path}/2017-2020_DF_9S.txt", header=0, usecols=usecols, skiprows=skiprows, nrows=nrows)

    folder_path = f"{output_dir}/predicted_results/"
    df0 = pd.read_csv(f"{folder_path}{input_station}_{model_type}_{feature_type}_{input_component}_{training_or_testing}_output.txt", header=0)

    date = np.array(df0.iloc[:, 0])
    obs_y_label = np.array(df0.iloc[:, 1])
    pre_y_label = np.array(df0.iloc[:, 2])

    failure_detected = 0
    success_detected = 0
    detected_status = []
    detected_status = []

    for step in range(len(df1)):
        id1 = df1.iloc[step, 0]
        id2 = df1.iloc[step, 1]

        id1 = np.where(date == id1)[0][0]
        id2 = np.where(date == id2)[0][0]

        pre_y_label_step = np.sum(pre_y_label[id1:id2])

        if pre_y_label_step == 0:
            failure_detected += 1
            detected_status.append(0)
            detected_status.append(0)
            print(f"{input_station}, {model_type}, {feature_type}, {training_or_testing}, {input_component}, "
                  f"failure_detected event, {date[id1]}, {date[id2]}")
        else:
            success_detected += 1
            detected_status.append(1)
            detected_status.append(1)

    cm_raw = confusion_matrix(obs_y_label, pre_y_label)
    f1 = f1_score(obs_y_label, pre_y_label, average='binary', zero_division=0)

    temp_dir = f"{output_dir}/results_summary"
    os.makedirs(name=temp_dir, exist_ok=True)
    f = open(f"{temp_dir}/summary_{training_or_testing}.txt", 'a')
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record = f"UTC+0, {now}, " \
             f"{input_station}, {model_type}, {feature_type}, {num_feats}, {training_or_testing}, {input_component}, " \
             f"TN, {cm_raw[0, 0]}, FP, {cm_raw[0, 1]}, " \
             f"FN, {cm_raw[1, 0]}, TP, {cm_raw[1, 1]}, F1, {f1:.4}," \
             f"total_event, {len(df1)}, failure_detected, {failure_detected}, success_detected, {success_detected}, " \
             f"detected_status, {detected_status}"
    f.write(str(record) + "\n")
    f.close()
