#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-02-23
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission

import os
import sys

import numpy as np
import numpy.typing as npt

from typing import Optional


import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier


# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path
current_dir = Path(__file__).resolve().parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent
import sys
sys.path.append(str(project_root))
# </editor-fold>


# import CONFIG_dir as a global variable
from config.config_dir import CONFIG_dir


def _save_output(be_saved_array, training_or_testing, output_dir, output_format):
    # make sure the column [:, 3] is predicted probability
    be_saved_array[:, 3] = np.round(be_saved_array[:, 3], 3)

    time_stamps_float = be_saved_array[:, 0]
    time_stamps_string = [datetime.utcfromtimestamp(ts).strftime("%Y-%m-%dT%H:%M:%S") for ts in time_stamps_float]
    time_stamps_string = np.array(time_stamps_string).reshape(-1, 1)

    save_output = np.hstack((time_stamps_string, be_saved_array))

    np.savetxt(f"{output_dir}/{output_format}-{training_or_testing}-output-{save_output[0, 0][:10]}.txt",
               save_output, delimiter=',', fmt='%s', comments='',
               header="time_window_start,time_stamps,obs_y_pro,obs_y_label,pre_y_pro,pre_y_label")

    return save_output



def ensemble_model(data_train, y_train, X_test, y_test, input_station, model_type, feature_type, input_component):

    # <editor-fold desc="add the sys.path to search for custom modules">
    from pathlib import Path
    current_dir = Path(__file__).resolve().parent
    # using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
    project_root = current_dir.parent
    import sys
    sys.path.append(str(project_root))
    # </editor-fold>

    model = "define it"
    if model_type == "Random_Forest":
        model = RandomForestClassifier(n_estimators=500)
    elif model_type == "XGBoost":
        model = XGBClassifier(n_estimators=500, importance_type="gain")
    else:
        print(f"check the model_type {model_type}")

    # model training
    model.fit(data_train, y_train)

    pre_y_train_label = model.predict(data_train) # model predicted data_train label
    pre_y_train_pro = model.predict_proba(data_train)[:, 1]  # model predicted data_train probability, column 0 is the pro of Noise
    pre_y_train_pro = np.round(pre_y_train_pro, decimals=3)

    # model testing
    pre_y_test_label = model.predict(X_test)
    pre_y_test_pro = model.predict_proba(X_test)[:, 1]
    pre_y_test_pro = np.round(pre_y_test_pro, decimals=3)

    # save mdoel parameters
    # the first is for reducing feature 1 by 1
    # feature_num = data_train.shape[1] # number of used features
    # joblib.dump(model, f"{CONFIG_dir['output_dir']}/train_test_output/trained_model_{feature_num}/{input_station}_{model_type}_{feature_type}_{input_component}.pkl")
    joblib.dump(model, f"{project_root}/trained_model/{input_station}_{model_type}_{feature_type}_{input_component}.pkl")

    # you can load the model as
    # model = joblib.load(f"{project_root}/output/trained_model/{input_station}_{model_type}_{feature_type}_{input_component}.pkl")

    return pre_y_train_label, pre_y_train_pro, pre_y_test_label, pre_y_test_pro, model


def ensemble_model_dual_test(X_test, ref_station, model_type, feature_type, ref_component):

    # you can load the reference model as
    model = joblib.load(f"{project_root}/trained_model/trained_model/{ref_station}_{model_type}_{feature_type}_{ref_component}.pkl")

    # model testing
    pre_y_test_label = model.predict(X_test)
    pre_y_test_pro = model.predict_proba(X_test)[:, 1]
    pre_y_test_pro = np.round(pre_y_test_pro, decimals=3)


    return pre_y_test_label, pre_y_test_pro


class ensemble_tree_model:
    def __init__(self,
                 #
                 params: str, # as ""
                 # data
                 train_data,
                 test_data,

                 # model
                 model_type: str,
                 n_estimators: int = 500,
                 class_weight = 1,

                 ) -> None:

        self.seismic_network, self.input_year, self.input_station, self.input_component = params.split("-")

        # set class output dir
        output_dir = f"{project_root}/trained_model"
        os.makedirs(output_dir, exist_ok=True)
        self.project_root = output_dir
        self.output_format = f"{params}-{class_weight}"


        # the train and test data must as
        # [row in time, time_stamps + features + label]
        self.data_train = train_data
        self.data_test = test_data

        # initial the model
        if model_type == "Random_Forest":
            # https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.RandomForestClassifier.html
            self.model = RandomForestClassifier(n_estimators=n_estimators,
                                                class_weight=class_weight)
        elif model_type == "XGBoost":
            # https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRegressor
            self.model = XGBClassifier(n_estimators=n_estimators,
                                       sample_weight=class_weight,
                                       importance_type="gain")
        elif model_type == "load_trained_model":
            self.model = joblib.load(f"{path}.pkl")
        else:
            print(f"check the model_type {model_type}")


    def sample_data(self, data, event_to_noise_ratio):
        '''
        sample the data to match DF:Non-DF = 1:event_to_noise_ratio
        Args:
            data: either self.train_data or self.test_data
            event_to_noise_ratio: int, equal to Non-DF/DF

        Returns:
            sampled data
        '''

        label_1_data = data[data[:, -1] == 1]  # DF
        label_0_data = data[data[:, -1] == 0]  # non-DF

        num_label_1 = len(label_1_data)
        num_label_0 = num_label_1 * event_to_noise_ratio
        assert num_label_0 > len(label_0_data), f"num_label_0 > len(label_0_data) in funcs. sample_data"

        sampled_label_1 = label_1_data
        sampled_label_0 = label_0_data[np.random.choice(len(label_0_data), num_label_0, replace=False)]

        temp = np.vstack((sampled_label_1, sampled_label_0))
        # sort as time series order
        temp = temp[temp[:, 0].argsort()]

        return temp

    def dump_results(self, purpose, t_target, target, obs_y_label, predicted_pro, pre_y_label):

        be_saved_array = np.cat((t_target.view(-1, 1),
                                 target.view(-1, 1),
                                 obs_y_label.view(-1, 1),
                                 predicted_pro.view(-1, 1),
                                 pre_y_label.view(-1, 1)), dim=1)  # as column

        _save_output(be_saved_array, purpose, self.project_root, self.output_format)

    def training(self, dump_model=True, purpose="training"):

        t_target = self.data_train[:, 0]
        features = self.data_train[:, 1:-1]
        target = self.data_train[:, -1]

        self.model.fit(features, target)

        # model predicted data_train probability, column 0 is the pro of non-DF
        predicted_pro = self.model.predict_proba(features)
        predicted_pro = np.round(predicted_pro, decimals=3)
        # model predicted data_train label
        pre_y_label = self.model.predict(features)

        # dump the results
        obs_y_label = target
        dump_results(purpose, t_target, target, obs_y_label, predicted_pro, pre_y_label)

        if dump_model is True:
            joblib.dump(self.model, f"{self.project_root}/{self.output_format}-trained.pkl")

        return

    def testing(self, purpose="testing"):

        t_target = self.data_test[:, 0]
        features = self.data_test[:, 1:-1]
        target = self.data_test[:, -1]

        # model predicted data_train probability, column 0 is the pro of non-DF
        predicted_pro = self.model.predict_proba(features)
        predicted_pro = np.round(predicted_pro, decimals=3)
        # model predicted data_train label
        pre_y_label = self.model.predict(features)

        # dump the results
        obs_y_label = target
        dump_results(purpose, t_target, target, obs_y_label, predicted_pro, pre_y_label)


    def activation(self, event_to_noise_ratio=None):

        if sample_data is True:
            self.data_train = self.sample_data(self.data_train, event_to_noise_ratio)
            self.data_test = self.sample_data(self.data_test, event_to_noise_ratio)
        else:
            pass

        self.training()
        self.testing()


