#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-02-23
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission

import os
from datetime import datetime

import pandas as pd
import numpy as np

from typing import Optional, Dict, List


import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier


# <editor-fold desc="define the parent directory">
import platform
if platform.system() == 'Darwin':
    parent_dir = "/"
elif platform.system() == 'Linux':
    parent_dir = "/home/qizhou/3paper/2ML-BL-enhance-DF-EWs"
else:
    print(f"check the parent_dir for platform.system() == {platform.system()}")
# add the parent_dir to the sys
import sys
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
else:
    pass
# </editor-fold>


# import CONFIG_dir as a global variable
from config.config_dir import CONFIG_dir
from functions.public.archive_data import dump_model_prediction, dump_evaluate_matrix
from functions.public.undersamp_training_data import under_sample_array

class Ensemble_Tree_Model:
    def __init__(self,
                 # data
                 train_data: np.ndarray,  # 2D numpy array
                 test_data: np.ndarray,  # 2D numpy array
                 input_features_name: List[str], # used feature name

                 # input_format
                 input_format: str = "9S-2017_2019-ILL12-EHZ-C",

                 # model
                 model_type: str = "XGBoost",
                 n_estimators: int = 500,

                 # Non-DF label : DF label = 1-class_weight : class_weight
                 class_weight: float = 0.9,  # weight for DF label
                 noise2event_ratio: int = 1e5 # either sample the training data or use all training data
                 ) -> None:


        # <editor-fold desc="set class output dir and format">
        output_dir = f"{CONFIG_dir['output_dir2']}/{model_type}"
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.output_format = input_format
        # </editor-fold>


        # <editor-fold desc="set dataset">
        # the train and test data must as
        # [row in time, time_stamps + features + label]
        self.data_train = train_data.astype(float)
        self.data_test = test_data.astype(float)
        self.input_features_name = input_features_name

        self.class_weight = class_weight
        self.noise2event_ratio = noise2event_ratio
        # </editor-fold>


        # <editor-fold desc="initial the model">
        if model_type == "Random_Forest":
            # https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.RandomForestClassifier.html
            # impurity-based feature importances
            self.model = RandomForestClassifier(n_estimators=n_estimators)
        elif model_type == "XGBoost":
            # https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRegressor
            # gain-based feature importances
            self.model = XGBClassifier(n_estimators=n_estimators,
                                       importance_type="gain")
        elif model_type == "load_trained_model":
            self.model = joblib.load(f"{self.output_dir}/{self.output_format}-trained-model.pkl")
        else:
            print(f"check the model_type {model_type}")
        # </editor-fold>

    def dump_feature_imp(self):

        imp = self.model.feature_importances_

        # normalize to [0, 1]
        imp_min = imp.min()
        imp_max = imp.max()
        imp = (imp - imp_min) / (imp_max - imp_min)

        temp = np.vstack((self.input_features_name, imp))

        with open(f"{self.output_dir}/{self.output_format}-feature-imp.txt", "a") as f:
            np.savetxt(f, temp, fmt='%s', delimiter=',')

    def training(self, dump_model=False, purpose="training"):

        t_target = self.data_train[:, 0]
        features = self.data_train[:, 1:-1]
        target = self.data_train[:, -1]

        # assign the weight for each sample
        sample_weights = np.where(target == 0, 1-self.class_weight, self.class_weight)
        self.model.fit(features, target, sample_weight=sample_weights)

        # model predicted data_train probability, column 0 is the pro of non-DF
        predicted_pro = self.model.predict_proba(features)[:, 1]
        predicted_pro = np.round(predicted_pro, decimals=3).astype(t_target.dtype)
        # model predicted data_train label
        pre_y_label = self.model.predict(features).astype(t_target.dtype)

        # dump the results
        obs_y_label = target
        be_saved_array = np.concatenate((t_target.reshape(-1, 1),
                                         target.reshape(-1, 1),
                                         obs_y_label.reshape(-1, 1),
                                         predicted_pro.reshape(-1, 1),
                                         pre_y_label.reshape(-1, 1)), axis=1)  # as column

        dump_model_prediction(be_saved_array, purpose, self.output_dir, self.output_format)
        dump_evaluate_matrix(be_saved_array, f"{purpose}, {self.output_format}",
                             f"{CONFIG_dir['output_dir2']}", "summary_ensemble_1by1")

        if dump_model is True:
            joblib.dump(self.model, f"{self.output_dir}/{self.output_format}-trained-model.pkl")

    def testing(self, purpose="testing"):

        t_target = self.data_test[:, 0]
        features = self.data_test[:, 1:-1]
        target = self.data_test[:, -1]

        # model predicted data_train probability, column 0 is the pro of non-DF
        predicted_pro = self.model.predict_proba(features)[:, 1]
        predicted_pro = np.round(predicted_pro, decimals=3).astype(t_target.dtype)
        # model predicted data_train label
        pre_y_label = self.model.predict(features).astype(t_target.dtype)

        # dump the results
        obs_y_label = target
        be_saved_array = np.concatenate((t_target.reshape(-1, 1),
                                         target.reshape(-1, 1),
                                         obs_y_label.reshape(-1, 1),
                                         predicted_pro.reshape(-1, 1),
                                         pre_y_label.reshape(-1, 1)), axis=1)  # as column

        dump_model_prediction(be_saved_array, purpose, self.output_dir, self.output_format)
        dump_evaluate_matrix(be_saved_array, f"{purpose}, {self.output_format}",
                             f"{CONFIG_dir['output_dir2']}", "summary_ensemble_1by1")

    def activation(self, training=True, testing=True, dump_imp=False):

        if self.noise2event_ratio < 300:
            # sample the training data, and make sure "Non-DF : DF = noise2event_ratio : 1"
            self.data_train = under_sample_array(self.data_train, self.noise2event_ratio)
        else:
            # use all training data
            pass

        if training is True:
            self.training()

        if dump_imp is True:
            self.dump_feature_imp()

        if testing is True:
            self.testing()

        return self.model
