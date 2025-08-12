#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-02-23
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission

import argparse
import os
from datetime import datetime

import numpy as np

# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path
current_dir = Path(__file__).resolve().parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent
import sys
sys.path.append(str(project_root))
# </editor-fold>



from functions.dataset2dataloader import *
from functions.results_achive import *
from functions.results_visualization import *
from functions.tree_ensemble_model import *
from functions.check_undetected_events import *
from functions.feature_imp_shap import *


def main(input_station, model_type, feature_type, input_component):
    print(f"Start Job: UTC+0, {input_station, model_type, feature_type, input_component}", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "\n")

    data_normalize = True

    # load data
    input_features_name, X_train, y_train, _, time_stamps_train = select_features(input_station, feature_type, input_component, "training", [2017, 2018, 2019])
    input_features_name, X_test,  y_test,  _, time_stamps_test =  select_features(input_station, feature_type, input_component, "testing", [2020])

    if data_normalize is True:
        X_train, X_test = input_data_normalize(X_train, X_test)
    else:
        pass

    # pass data to model
    pre_y_train_label, pre_y_train_pro, \
    pre_y_test_label,  pre_y_test_pro, \
    model = ensemble_model(X_train, y_train, X_test, y_test, input_station, model_type, feature_type, input_component)

    # achieve the results
    achieve_predicted_results(time_stamps_train, y_train, pre_y_train_label, pre_y_train_pro,
                              input_station, model_type, feature_type, input_component, "training")

    achieve_predicted_results(time_stamps_test, y_test, pre_y_test_label, pre_y_test_pro,
                              input_station, model_type, feature_type, input_component, "testing")

    # vasulize the confusion_matrix
    visualize_confusion_matrix(y_train, pre_y_train_label, "training",
                               input_station, model_type, feature_type, input_component)

    visualize_confusion_matrix(y_test, pre_y_test_label, "testing",
                               input_station, model_type, feature_type, input_component)

    # summary the results
    summary_results(input_station, model_type, feature_type, input_component, "training")
    summary_results(input_station, model_type, feature_type, input_component, "testing")

    # vasulize the feature importance
    if feature_type == "C":
        imp = model.feature_importances_
        visualize_feature_imp("build_in", imp, input_features_name,
                              input_station, model_type, feature_type, input_component)

        #imp = shap_imp(input_station, model_type, feature_type, input_component, X_train)
        #visualize_feature_imp("shap_value", imp, input_features_name,
                              #input_station, model_type, feature_type, input_component)
    else:
        pass

    print(f"End Job: UTC+0, {input_station, model_type, feature_type, input_component}", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='input parameters')

    parser.add_argument("--input_station", default="ILL12", type=str, help="input station")
    parser.add_argument("--model_type", default="Random_Forest", type=str, help="model type")
    parser.add_argument("--feature_type", default="C", type=str, help="feature type")
    parser.add_argument("--input_component", default="EHZ", type=str, help="seismic input_component")

    args = parser.parse_args()

    main(args.input_station, args.model_type, args.feature_type, args.input_component)

