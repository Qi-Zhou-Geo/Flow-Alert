#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-02-23
#__author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
#__find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission

import os
import argparse

# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path
current_dir = Path(__file__).resolve().parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys
sys.path.append(str(project_root))
# </editor-fold>

# import the custom functions
from functions.public.dataset_to_dataloader import *
from functions.model.tree_ensemble_model import Ensemble_Tree_Model

def prepare_dataloader(feature_type, params, repeat=1):
    # empty list to store the dataloader
    train_dataloader = []
    test_dataloader = []

    # loop the input data-60s
    for p in params:

        seismic_network, input_year, input_station, input_component, dataloader_type, with_label = p.split("-")

        # convert str to bool
        if with_label == "True":
            with_label = True
        else:
            with_label = False

        # load data_array as [time_stamps, features, target]
        input_features_name, data_array = select_features(catchment_name,
                                                          seismic_network,
                                                          input_year,
                                                          input_station,
                                                          input_component,
                                                          feature_type,
                                                          with_label,
                                                          repeat=repeat)

        # convert data-60s frame to data-60s loader
        if dataloader_type == "training":
            train_dataloader.append(data_array)
        elif dataloader_type == "testing":
            test_dataloader.append(data_array)
        else:
            print(f"check the dataloader_type={dataloader_type}")

    # stack by row
    train_data, test_data = np.vstack(train_dataloader), np.vstack(test_dataloader)

    return input_features_name, train_data, test_data


def main(model_type, feature_type, class_weight, noise2event_ratio, params, repeat):

    job_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    time_now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    print(f"Start Job={job_id}, UTC+0={time_now}, "
          f"{model_type, feature_type, class_weight, noise2event_ratio}", "\n")

    # load the train_data and test-data-60s
    input_features_name, train_data, test_data = prepare_dataloader(feature_type, params, repeat=repeat)
    print(f"train_data.shape, {train_data.shape}, train DF, {np.sum(train_data[:, -1])}, "
          f"test_data.shape, {test_data.shape}, test DF, {np.sum(test_data[:, -1])}")

    # train or test class
    input_format = f"{params[0]}-repeat-{repeat}-{model_type}-{feature_type}-DFweight-{class_weight}-ratio-{noise2event_ratio}"
    n_estimators = 500
    workflow = Ensemble_Tree_Model(train_data, test_data, input_features_name,
                                   input_format, model_type, n_estimators,
                                   class_weight, noise2event_ratio)

    if feature_type == "C": # dump the feature imp
        workflow.activation(dump_imp=True)
    else:
        workflow.activation(dump_imp=False)

    time_now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    print(f"End Job={job_id}, UTC+0={time_now}, "
          f"{model_type, feature_type, class_weight, noise2event_ratio}", "\n")


if __name__ == "__main__":
    # sinfo -n node[501-514] -N --Format="Nodelist,CPUsState,AllocMem,Memory,GresUsed,Gres"
    parser = argparse.ArgumentParser(description='input parameters')

    parser.add_argument("--model_type", default="XGBoost", type=str, help="model type")
    parser.add_argument("--feature_type", default="C", type=str, help="feature type")

    parser.add_argument("--class_weight", default=0.9, type=float, help="weight for DF label")
    parser.add_argument("--noise2event_ratio", default=1, type=int, help="Non-DF to DF label ratio")

    parser.add_argument("--params", nargs='+', type=str, help="list of stations")

    parser.add_argument("--num_repeat", default=6, type=int, help="num of repeat")

    args = parser.parse_args()

    for repeat in range(1, args.num_repeat): # repate 5 times
        main(args.model_type, args.feature_type,
             args.class_weight, args.noise2event_ratio,
             args.params, repeat)
