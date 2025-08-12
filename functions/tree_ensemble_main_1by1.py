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

from dataset2dataloader import *
from results_achive import *
from results_visualization import *
from tree_ensemble_model import *
from check_undetected_events import *
from feature_imp_shap import *

def main(input_station, model_type, feature_type, input_component, idxs):
    print(f"Start Job: UTC+0, {input_station, model_type, feature_type, input_component}", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "\n")
    print(f"Indexes (in main): {idxs}")
    num_idxs = len(idxs)
    data_normalize = True

    # load data
    input_features_name1, X_train, y_train, _, time_stamps_train = select_features(input_station, feature_type, input_component, "training")
    input_features_name1, X_test,  y_test,  _, time_stamps_test =  select_features(input_station, feature_type, input_component, "testing")
    # select specific features
    X_train, input_features_name = select_specific_features(X_train, input_features_name1, idxs)
    X_test, input_features_name = select_specific_features(X_test, input_features_name1, idxs)
    

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
                              input_station, model_type, feature_type, input_component, "training", num_idxs)

    achieve_predicted_results(time_stamps_test, y_test, pre_y_test_label, pre_y_test_pro,
                              input_station, model_type, feature_type, input_component, "testing", num_idxs)

    # vasulize the confusion_matrix
    visualize_confusion_matrix(y_train, pre_y_train_label, "training",
                               input_station, model_type, feature_type, input_component, num_idxs)

    visualize_confusion_matrix(y_test, pre_y_test_label, "testing",
                               input_station, model_type, feature_type, input_component, num_idxs)

    # summary the results
    summary_results(input_station, model_type, feature_type, input_component, "training", num_idxs)
    summary_results(input_station, model_type, feature_type, input_component, "testing", num_idxs)

    # vasulize the feature importance
    if feature_type == "C":
        imp = model.feature_importances_
        visualize_feature_imp("build_in", imp, input_features_name,
                              input_station, model_type, feature_type, input_component, num_idxs)

        imp = shap_imp(input_station, model_type, feature_type, input_component, X_train, num_idxs)
        visualize_feature_imp("shap_value", imp, input_features_name,
                              input_station, model_type, feature_type, input_component, num_idxs)
    else:
        pass

    print(f"End Job: UTC+0, {input_station, model_type, feature_type, input_component}", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='input parameters')

    parser.add_argument("--input_station", default="ILL12", type=str, help="input station")
    parser.add_argument("--model_type", default="Random_Forest", type=str, help="model type")
    parser.add_argument("--feature_type", default="C", type=str, help="feature type")
    parser.add_argument("--input_component", default="EHZ", type=str, help="seismic input_component")
    parser.add_argument("--indexes", default='', type= str, help= "list of input features")

    args = parser.parse_args()

    print(args.indexes)
    indexes = list(map(int, args.indexes.split(',')))
    print(indexes)

    dir_list = [f'/home/kshitkar/ML-BL-enhance-DF-EWs/output/figures_{len(indexes)}',
                f'/home/kshitkar/ML-BL-enhance-DF-EWs/output/trained_model_{len(indexes)}',
                f'/home/kshitkar/ML-BL-enhance-DF-EWs/output/predicted_results_{len(indexes)}']
    try:
        for output_dir in dir_list:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
    except FileExistsError:
        pass
    
    main(args.input_station, args.model_type, args.feature_type, args.input_component, indexes)

