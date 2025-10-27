#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-02-23
#__author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
#__find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission

import os
import argparse

import pandas as pd
import numpy as np

from obspy import UTCDateTime

import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path

import pandas as pd

current_dir = Path(__file__).resolve().parent

# using ".parent" on "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys
sys.path.append(str(project_root))
# </editor-fold>

# import the custom functions
# from functions.model.tree_ensemble_model import Ensemble_Tree_Model
from functions.data_process.archive_data import dump_model_prediction, dump_evaluate_matrix
from functions.warning_strategy.buffer_prediction import cal_buffered_cm



def load_H_features(model_version="v1dot3model", training_testing_seperator="2020-05-29T00:00:00"):

    df = pd.read_csv(f"{project_root}/pipeline/{model_version}/"
                     f"train_test_2017-2020_9repeat/LSTM/input_features_H.txt",
                     header=0)

    date_time = df["time_window_start"].values # str time stamps
    id = np.where(date_time == training_testing_seperator)[0][0]

    train_data = df.iloc[:id, 1:] # with float time
    input_features_name = train_data.iloc[:, 1:-1].columns.values
    test_data = df.iloc[id:, 1:]

    train_data, test_data = np.array(train_data), np.array(test_data)

    return input_features_name, train_data, test_data


def workflow(model, model_type, t_target, features, target, output_path, output_name, purpose, repeat):

    predicted_pro = model.predict_proba(features)[:, 1]
    pre_y_pro = np.round(predicted_pro, decimals=3).astype(t_target.dtype)
    # model predicted train_data label
    pre_y_label = model.predict(features).astype(t_target.dtype)

    time_window_start = np.array([UTCDateTime(i).isoformat() for i in t_target])
    obs_y_pro = np.full(len(t_target), np.nan)
    obs_y_label = target
    be_saved_array = np.concatenate((time_window_start.reshape(-1, 1),
                                     t_target.reshape(-1, 1),
                                     obs_y_pro.reshape(-1, 1),
                                     obs_y_label.reshape(-1, 1),
                                     pre_y_pro.reshape(-1, 1),
                                     pre_y_label.reshape(-1, 1)), axis=1)  # as column
    temp_df = pd.DataFrame(be_saved_array, columns=["time_window_start", "time_stamps",
                                                    "obs_y_pro", "obs_y_label",
                                                    "pre_y_pro", "pre_y_label"])

    temp_df.to_csv(f"{output_path}/{output_name}_{purpose}_{repeat}.txt", index=False)

    # buffer the 5 minutes before and 60 minutes after the event
    print(obs_y_label, pre_y_label)
    cm_buffered, f1_buffered = cal_buffered_cm(obs_y_label=obs_y_label,
                                               pre_y_label=pre_y_label,
                                               buffer_l=5, buffer_r=60)
    tn, fp, fn, tp = cm_buffered.ravel()

    epsilon = 1e-8
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = f1_buffered
    f2 = tp / (tp + fp + 0.5 * fn + epsilon)

    temp = [tn, fp, fn, tp, precision, recall, f1, f2]
    temp = ", ".join(map(str, temp))
    time_now = UTCDateTime().isoformat()
    record = f"{model_type}, {time_now}, {purpose}, {repeat}, buffered_cm, tn, fp, fn, tp, precision, recall, f1, f2, {temp}\n"

    with open(f'{output_path}/tree_model_report.txt', 'a') as f:
        f.write(record)

def model_train_test(train_data, test_data, model_type, n_estimators, class_weight, output_path, output_name, repeat):

    if model_type in ["Random_Forest", "RF"]:
        # https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        # impurity-based feature importances
        model = RandomForestClassifier(n_estimators=n_estimators)
    elif model_type in ["XGBoost", "XGB"]:
        # https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRegressor
        # gain-based feature importances
        model = XGBClassifier(n_estimators=n_estimators, importance_type="gain")
    else:
        print(f"check the model_type {model_type}")


    # train the model
    purpose = "training"

    # give label 0 with weight "class_weight" and 1 with "1 - class_weight"
    t_target = train_data[:, 0]
    features = train_data[:, 1:-1]
    target = train_data[:, -1]

    sample_weights = np.where(target == 0, 1 - class_weight, class_weight)
    model.fit(features, target, sample_weight=sample_weights)

    if model_type in ["Random_Forest", "RF"]:
        joblib.dump(model, f"{output_path}/Model=RF_STA=ILL02_Feature=H_repeat={repeat+1}.joblib", compress=9)
    elif model_type in ["XGBoost", "XGB"]:
        model.save_model(f"{output_path}/Model=XGB_STA=ILL02_Feature=H_repeat={repeat+1}.ubj")
    else:
        print(f"check the model_type {model_type}")

    # run the workflow
    workflow(model, model_type, t_target, features, target, output_path, output_name, purpose, repeat)

    # test the model
    purpose = "testing"
    
    t_target = test_data[:, 0]
    features = test_data[:, 1:-1]
    target = test_data[:, -1]
    
    # run the workflow
    workflow(model, model_type, t_target, features, target, output_path, output_name, purpose, repeat)


def main(model_type, n_estimators=500, class_weight=0.9, num_repeat=3):

    time_now = UTCDateTime().isoformat()
    print(f"UTC+0={time_now}, {model_type,  class_weight}", "\n")

    # load the train data and test data as [t_target, features, target]
    input_features_name, train_data, test_data = load_H_features(training_testing_seperator="2020-05-29T00:00:00")
    print(f"train_data.shape, {train_data.shape}, train DF, {np.sum(train_data[:, -1])}, "
          f"test_data.shape, {test_data.shape}, test DF, {np.sum(test_data[:, -1])}")

    # set the output path
    output_path = f"{current_dir}/train_test_2017-2020_1repeat_tree"
    os.makedirs(output_path, exist_ok=True)
    output_name = f"Illgraben-9S-2017-ILL02-EHZ-H-training-True-repeat-1-{model_type}-H"

    # run the train test
    for repeat in range(num_repeat):
        model_train_test(train_data, test_data, model_type, n_estimators, class_weight, output_path, output_name, repeat)
    
        time_now = UTCDateTime().isoformat()
        print(f"UTC+0={time_now}, {repeat, model_type,  class_weight}", "\n")



if __name__ == "__main__":
    # sinfo -n node[501-514] -N --Format="Nodelist,CPUsState,AllocMem,Memory,GresUsed,Gres"
    parser = argparse.ArgumentParser(description='input parameters')
    parser.add_argument("--model_type", default="XGB", type=str)
    args = parser.parse_args()

    main(args.model_type)