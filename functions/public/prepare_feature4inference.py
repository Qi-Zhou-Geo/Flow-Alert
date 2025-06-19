#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2025-02-17
#__author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission

import sys
import yaml

import pytz
from datetime import datetime, timedelta

from tqdm import tqdm

import numpy as np
import pandas as pd

from obspy import read, Stream, Trace, read_inventory, signal
from obspy.core import UTCDateTime # default is UTC+0 time zone

# import the custom functions
from calculate_features.Type_A_features import calBL_feature                 # import Qi's all features (by *)
from calculate_features.Type_B_features import calculate_all_attributes      # import Clement's all features (by *)
from functions.public.min_max_normalize_transformer import min_max_normalize
from functions.seismic_data_processing_obspy.remove_outlier import smooth_outliers

class Stream_to_feature:
    def __init__(self, sub_window_size, window_overlap):

        self.sub_window_size = sub_window_size
        self.window_overlap = window_overlap

    def trim_st(self, st):
        # trim the Stream to get the minutes level (remove the seconds)

        if type(st) is Stream:
            st.merge(method=1, fill_value='latest', interpolation_samples=0)
            st._cleanup()
            tr = st[0]
        elif type(st) is Trace:
            tr = st

        # make sure the seismic trace with inter "00" at second level
        if tr.stats.starttime.strftime("%Y-%m-%dT%H:%M:%S")[-2:] != "00":
            d1 = tr.stats.starttime.strftime("%Y-%m-%dT%H:%M")
            tr = tr.trim(starttime=UTCDateTime(d1) + 60,
                         endtime=tr.stats.endtime, nearest_sample=False)

        if tr.stats.endtime.strftime("%Y-%m-%dT%H:%M:%S")[-2:] != "00":
            d2 = tr.stats.endtime.strftime("%Y-%m-%dT%H:%M")
            tr = tr.trim(starttime=tr.stats.starttime,
                         endtime=UTCDateTime(d2) - 60, nearest_sample=False)

        return tr # Trace

    def cal_attributes_A(self, data_array, scaling=1e9, ruler=300):  # the main function is from Qi
        data_array_nm = data_array * scaling  # converty m/s to nm/s
        feature_array = calBL_feature(data_array_nm, ruler)

        # first digits frequency 1-9, goodness of fit, and alpha
        selected = np.arange(0, 9).tolist() # first digit frequency
        selected.append(10) # goodness of fit
        selected.append(13) # alpha
        feature_array = feature_array[selected]

        return feature_array  # 17 features

    def cal_attributes_B(self, data_array, sps):  # the main function is from Clement
        # sps: sampling frequency; flag=0: one component seismic signal;
        features = calculate_all_attributes(Data=data_array, sps=sps, flag=0)[0]  # feature 1 to 60
        feature_array = features[1:]  # leave features[0]=event_duration

        return feature_array  # 59 features

    def chunk_data(self, st_data, sub_window_size, window_overlap, st_startime_array, st_end_time, st_sps, npts):

        '''
        Chunk the data-60s
        Args:
            st: Obspy Trace
            sub_window_size: int or float, unit by second
            window_overlap: float value in [0, 1], 0 means no overlap

        Returns:

        '''

        chunk_length = int(st_sps * sub_window_size)  # unit by data-60s point
        step_size = int(chunk_length - chunk_length * window_overlap) # unit by data-60s point

        # chunk the data-60s
        chunk_x = np.lib.stride_tricks.sliding_window_view(st_data, window_shape=chunk_length)
        chunk_x = chunk_x[::step_size]

        chunk_t = np.linspace(st_startime_array, st_end_time, npts)
        chunk_t = np.lib.stride_tricks.sliding_window_view(chunk_t, window_shape=chunk_length)
        chunk_t = chunk_t[::step_size]
        chunk_t = chunk_t[:, 0].reshape(-1)
        chunk_t = chunk_t.astype(float)

        chunk_t_str = [datetime.fromtimestamp(i, tz=pytz.utc).strftime('%Y-%m-%dT%H:%M:%S.%f') for i in chunk_t]
        chunk_t_str = np.array(chunk_t_str)

        return chunk_t_str, chunk_t, chunk_x

    def normalize_feature(self, output_feature, normalize_type):

        if normalize_type == "ref-training":

            input_station, feature_type = "ILL02", "D"
            df = pd.DataFrame(output_feature)
            df.iloc[:, 2:] = min_max_normalize(df.iloc[:, 2:],
                                               input_station="ILL02",
                                               feature_type=feature_type)
            output_feature = np.array(df)
        elif normalize_type == "ref-itself":

            feature = output_feature[:, 2:].astype(float)
            feature = smooth_outliers(feature, row_or_column="column",
                                      outliers=95, smooth_window=1,
                                      replace_method="Q95")

            feature_std = (feature - feature.min(axis=0)) / (feature.max(axis=0) - feature.min(axis=0) + 1e-6)
            # scale to [min_val, max_val], Min-Max normalize for each column (feature) in time
            min_val, max_val = 0, 1
            feature_scaled = feature_std * (max_val - min_val) + min_val

            output_feature[:, 2:] = feature_scaled

        return output_feature

    def prepare_feature(self, st, num_features=70, print_reminder=True, normalize_type="ref-training"):

        tr = self.trim_st(st)
        if print_reminder is True:
            print("This action <prepare_feature> will take a while, be patient.")

        st_data = tr.data
        st_startime_array = tr.stats.starttime.timestamp
        st_end_time = tr.stats.endtime.timestamp
        st_sps = tr.stats.sampling_rate
        npts = tr.stats.npts

        chunk_t_str, chunk_t, chunk_x = self.chunk_data(st_data,
                                                        self.sub_window_size,
                                                        self.window_overlap,
                                                        st_startime_array, st_end_time, st_sps, npts)

        st_sps = tr.stats.sampling_rate

        output_feature = np.empty((chunk_t_str.size, num_features+2), dtype=object) # with [t_str, t]
        for idx, (t_str, t_float, data) in tqdm(enumerate(zip(chunk_t_str, chunk_t, chunk_x)),
                                                total=len(chunk_t_str),
                                                desc="Progress of <prepare_feature>",
                                                file=sys.stdout):

            time_array = np.array([t_str, t_float])

            type_a = self.cal_attributes_A(data_array=data)
            type_b = self.cal_attributes_B(data_array=data, sps=st_sps) # without network features

            output_feature[idx, :] = np.concatenate((time_array, type_a, type_b), axis=0)

        if normalize_type is not None:
            output_feature = self.normalize_feature(output_feature, normalize_type)
            return output_feature
        elif normalize_type is None:
            return output_feature
        else:
            print(f"check the normalize_type {normalize_type}")
