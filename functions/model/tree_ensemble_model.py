#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-02-23
#__author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
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
from scipy.stats import t as student_t  # Student's t-distribution


# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path
current_dir = Path(__file__).resolve().parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys
sys.path.append(str(project_root))
# </editor-fold>

# import the custom functions
from functions.public.archive_data import dump_model_prediction, dump_evaluate_matrix
from functions.public.undersamp_training_data import under_sample_array

class Ensemble_Tree_Model:
    def __init__(self,
                 # data
                 train_data: np.ndarray,  # 2D numpy array
                 test_data: np.ndarray,  # 2D numpy array
                 input_features_name: List[str], # used feature name

                 # input_format
                 output_dir: str = None,
                 input_format: str = "9S-2017_2019-ILL12-EHZ-C",

                 # model
                 model_type: str = "XGBoost",
                 n_estimators: int = 500,

                 # Non-DF label : DF label = 1-class_weight : class_weight
                 class_weight: float = 0.9,  # weight for DF label
                 noise2event_ratio: int = 1e5 # either sample the training data or use all training data
                 ) -> None:


        # <editor-fold desc="set class output dir and format">
        if output_dir is None:
            current_dir = Path(__file__).resolve().parent
            project_root = current_dir.parent.parent
            output_dir = f"{project_root}/output"

        self.output_dir = f"{output_dir}/{model_type}"
        os.makedirs(self.output_dir, exist_ok=True)
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
                             f"{self.output_dir}", "summary_ensemble_1by1")

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
                             f"{self.output_dir}", "summary_ensemble_1by1")

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

class Ensemble_Trained_Tree_Classifier:
    def __init__(self, trained_model_name, model_version, model_dir=None, station=None):
        self.trained_model_name = trained_model_name
        self.model_version = model_version
        self.model_dir = model_dir
        self.station = station

    def load_trained_model(self, trained_model_name, repeate):
        """
        Load a pre-trained ensemble model (e.g., RF or XGB).
        """
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parent.parent

        if self.model_dir is None:
            ref_model_dir = Path(f"../../trained_model/{self.model_version}").resolve()
        else:
            ref_model_dir = Path(self.model_dir).resolve()

        model_path = f"{ref_model_dir}/{self.station}_{trained_model_name}_repeat{repeate}.pkl"
        model = joblib.load(model_path)
        print(f"Loaded pre-trained model: {model_path}")
        return model

    def ensemble_models(self, num_repeate=5):
        """
        Load multiple trained ensemble models for ensembling.
        """
        models = []
        for repeate in range(1, num_repeate + 1):
            model = self.load_trained_model(trained_model_name=self.trained_model_name, repeate=repeate)
            models.append(model)
        return models

    def statistical_testing(self, predicted_pro, row_or_column="column", confidence_level=0.95):
        """
        Perform statistical testing to calculate mean and confidence intervals.
        """
        predicted_pro = np.array(predicted_pro)

        if predicted_pro.ndim == 1:
            axis = 0
        elif predicted_pro.ndim == 2:
            axis = 1 if row_or_column == "row" else 0

        pro_mean = np.mean(predicted_pro, axis=axis)
        sem = np.std(predicted_pro, axis=axis, ddof=1) / np.sqrt(predicted_pro.shape[axis])
        degree_of_freedom = predicted_pro.shape[axis] - 1
        alpha = 1 - confidence_level
        tail = 1 - alpha / 2
        ci_range = student_t.ppf(tail, degree_of_freedom) * sem

        return pro_mean, ci_range

    def predictor_from_data(self, features, models):
        """
        Predict probabilities using ensemble models.
        """
        predicted_pro = np.empty((len(features), len(models)))

        for idx, model in enumerate(models):
            predicted_pro[:, idx] = model.predict_proba(features)[:, 1]

        pro_mean, ci_range = self.statistical_testing(predicted_pro, row_or_column="row")
        return predicted_pro, pro_mean, ci_range

    def predictor_from_dataloader(self, features, models):
        """
        Predict probabilities for a single sequence using ensemble models.
        """
        array_temp = np.empty((0, 4+len(models)))
        predicted_pro = np.empty((features.shape[0], len(models)))
        for idx, model in enumerate(models):
            t_target, input_features, target = features[:,0], features[:,1:-1], features[:,-1]
            pro = model.predict_proba(input_features)[:, 1]
            predicted_pro[:, idx] = np.round(pro.reshape(-1), 3) # keep 3 decimal places
        pro_mean, ci_range = self.statistical_testing(predicted_pro, row_or_column="row")
        record = np.concatenate((t_target.reshape(-1, 1),
                                target.reshape(-1, 1),
                                predicted_pro,
                                pro_mean.reshape(-1, 1),
                                ci_range.reshape(-1, 1)), axis=1)  # as column

        array_temp = np.vstack((array_temp, record))  # as row

        # predicted_pro = np.array(predicted_pro).flatten()
        # pro_mean, ci_range = self.statistical_testing(predicted_pro)
        # return predicted_pro.tolist(), pro_mean, ci_range
        sort_indices = array_temp[:, 0].argsort()
        array_temp = array_temp[sort_indices]

        return array_temp