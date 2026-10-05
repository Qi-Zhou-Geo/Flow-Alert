#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = Last modified: 2026-09-14T17:03:30
# __author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission

import yaml

from tqdm import tqdm
from multiprocessing import Pool

import numpy as np
import pandas as pd

from obspy import UTCDateTime  # default is UTC+0 time zone

# region ### add the sys.path to search for custom modules ###
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
current_dir = current_file.parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent

sys.path.append(str(project_root))
# endregion


# import the custom functions
from functions.feature.feature_name import load_feature_name
from functions.feature.calculator import cal_attributes_A, cal_attributes_B
from functions.seismic.chunk_st2seq import chunk_data
from functions.visualize.visualize_seismic import convert_st2tr
from functions.data_process.dynamic_normalize import dynamic_norm


# for multiple process
def _process_worker(args):
    idx, t_str, t_float, data, selected_id, st_sps, cal_attributes_A, cal_attributes_B = args

    time_array = np.array([t_str, t_float, 0])

    type_a = cal_attributes_A(data_array=data)
    type_a = type_a[selected_id]
    type_b = cal_attributes_B(data_array=data, sps=st_sps)
    type_b_net = np.random.rand(10)

    result = np.concatenate((type_a, type_b, type_b_net, time_array), axis=0)

    return idx, result


class Stream_to_feature_self:
    def __init__(self, sub_window_size, window_overlap, feature_type):

        self.sub_window_size = sub_window_size
        self.window_overlap = window_overlap
        self.feature_type = feature_type
        self.normalize = False

        # feature
        self.cal_attributes_A = cal_attributes_A
        self.cal_attributes_B = cal_attributes_B
        self.chunk_data = chunk_data

        # time format
        self.fmt = "%Y-%m-%dT%H:%M:%S"

    def trim_st(self, st):
        # trim the Stream to get the minutes level (remove the seconds)

        tr = convert_st2tr(st)

        # make sure the seismic trace with inter "00" at second level
        stats = tr.stats  # type: ignore
        if stats.starttime.strftime(self.fmt)[-2:] != "00":
            d1 = stats.starttime.strftime(self.fmt)
            tr = tr.trim(starttime=UTCDateTime(d1) + 60, endtime=stats.endtime, nearest_sample=False)

        if stats.endtime.strftime(self.fmt)[-2:] != "00":
            d2 = stats.endtime.strftime(self.fmt)
            tr = tr.trim(starttime=stats.starttime, endtime=UTCDateTime(d2) - 60, nearest_sample=False)

        return tr  # Trace

    def normalize_feature(self, output_feature):
        feature_Name_A, feature_Name_B = load_feature_name(print_log=False)

        selected_id = self.select_type_A()
        columns_name = (
            np.array(feature_Name_A[4:])[selected_id].tolist()
            + feature_Name_B[4:]
            + [f"useless{i}" for i in range(10)]
            + ["time_float", "time_str", "label"]
        )
        df = pd.DataFrame(output_feature, columns=columns_name)

        assert df.shape[1] == 83, f"{df.shape[1]} != 83"

        max_factor, min_factor = dynamic_norm(
            df=df,
            Illgraben_rms_min=1e-6,
            Illgraben_rms_max=5e-4,
            theory_min=1e-8,
            theory_max=1e-3,
            window_size=60,
        )

        X = df.iloc[:, :-3].to_numpy().astype(float)
        scaled = (X - min_factor) / (max_factor - min_factor)
        df[df.columns[:-3]] = scaled

        output_feature = np.array(df)

        return output_feature

    def selected_feature_by_type(self, feature_type, output_feature):

        config_path = f"{project_root}/config/config_inference.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        selected = config[f"feature_type_{feature_type}"]
        selected_id = selected + [80, 81, 82]
        selected_df = output_feature[:, selected_id]

        return selected_id, selected_df

    def select_type_A(self):
        # shape A features is 17
        # first digits frequency 1-9, goodness of fit, and alpha
        selected_id = np.arange(0, 9).tolist()  # first digit frequency
        selected_id.append(10)  # goodness of fit
        selected_id.append(13)  # alpha

        return selected_id

    def prepare_feature(self, st, num_features=80, print_reminder=True):

        tr = self.trim_st(st)
        if print_reminder is True:
            print("This action <prepare_feature> will take a while, be patient.")

        st_data = tr.data  # type: ignore
        st_startime_array = tr.stats.starttime.timestamp  # type: ignore
        st_end_time = tr.stats.endtime.timestamp  # type: ignore
        st_sps = tr.stats.sampling_rate  # type: ignore
        npts = tr.stats.npts  # type: ignore

        chunk_t_str, chunk_t, chunk_x = self.chunk_data(
            st_data,
            self.sub_window_size,
            self.window_overlap,
            st_startime_array,
            st_end_time,
            st_sps,
            npts,
        )

        st_sps = tr.stats.sampling_rate  # type: ignore
        selected_id = self.select_type_A()

        output_feature = np.empty((chunk_t_str.size, num_features + 3), dtype=object)  # with [t_str, t_float]
        for idx, (t_str, t_float, data) in tqdm(
            enumerate(zip(chunk_t_str, chunk_t, chunk_x)),
            total=len(chunk_t_str),
            desc="Progress of <prepare_feature>",
            file=sys.stdout,
        ):
            time_array = np.array([t_str, t_float, 0])  # the label alway set as 0 (Non-DF), do not need it

            type_a = self.cal_attributes_A(data_array=data)
            type_a = type_a[selected_id]
            type_b = self.cal_attributes_B(data_array=data, sps=st_sps)  # without network features
            type_b_net = np.random.rand(10)  # pretended network features, do not need it

            # the shape is 83
            output_feature[idx, :] = np.concatenate((type_a, type_b, type_b_net, time_array), axis=0)  # stack as column

        # normalize the features
        # input as 2D array ([float time stamps, all 80 features, labels])
        output_feature = self.normalize_feature(output_feature)

        # select the feature by deseried type,
        # output as 2D array ([float time stamps, selected features, labels])
        selected_id, output_feature = self.selected_feature_by_type(self.feature_type, output_feature)

        return output_feature

    def prepare_feature_mpi(self, st, num_features=80, print_reminder=True, num_cpus=6):

        tr = self.trim_st(st)
        if print_reminder is True:
            print("This action <prepare_feature> will take a while, be patient.")

        st_data = tr.data  # type: ignore
        st_startime_array = tr.stats.starttime.timestamp  # type: ignore
        st_end_time = tr.stats.endtime.timestamp  # type: ignore
        st_sps = tr.stats.sampling_rate  # type: ignore
        npts = tr.stats.npts  # type: ignore

        # chunk the time series data
        chunk_t_str, chunk_t, chunk_x = self.chunk_data(
            st_data, self.sub_window_size, self.window_overlap, st_startime_array, st_end_time, st_sps, npts
        )

        st_sps = tr.stats.sampling_rate  # type: ignore
        selected_id = self.select_type_A()

        # prepare arguments for parallel processing
        args_list = [
            (idx, t_str, t_float, data, selected_id, st_sps, self.cal_attributes_A, self.cal_attributes_B)
            for idx, (t_str, t_float, data) in enumerate(zip(chunk_t_str, chunk_t, chunk_x))
        ]

        # use multiple workers
        output_feature = np.empty((chunk_t_str.size, num_features + 3), dtype=object)

        with Pool(processes=num_cpus) as pool:
            # Use imap for progress bar
            results = list(
                tqdm(
                    pool.imap(_process_worker, args_list),
                    total=len(args_list),
                    desc="Progress of <prepare_feature>",
                    file=sys.stdout,
                )
            )

        # fill output array
        for idx, result in results:
            output_feature[idx, :] = result

        # normalize the features
        # input as 2D array ([float time stamps, all 80 features, labels])
        output_feature = self.normalize_feature(output_feature)

        # select the feature by deseried type,
        # output as 2D array ([float time stamps, selected features, labels])
        selected_id, output_feature = self.selected_feature_by_type(self.feature_type, output_feature)

        return output_feature
