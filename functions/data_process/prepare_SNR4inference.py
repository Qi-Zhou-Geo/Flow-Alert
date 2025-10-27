#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2025-02-17
#__author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission

import sys

from tqdm import tqdm

import numpy as np
import pandas as pd

from scipy.stats import skew

from obspy import read, Stream, Trace, read_inventory, signal

# import the custom functions
from functions.data_process.min_max_normalize_transformer import min_max_normalize
from functions.seismic.remove_outlier import smooth_outliers
from functions.seismic.chunk_st2seq import chunk_data
from functions.seismic.welch_spectrum import welch_psd

class Stream_to_matrix:
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

    def snr(self, seismic_data_seq, method="max-mean"):
        # same as Eseis signal_snr method.
        # https://rdrr.io/cran/eseis/man/signal_snr.html

        if method == "max-mean":
            t_snr = np.max(seismic_data_seq) / np.mean(seismic_data_seq)
        else:
            t_snr = np.mean(seismic_data_seq) / np.std(seismic_data_seq)

        return t_snr


    def skewness(self, seismic_data_seq, sampling_freq, f_min=1, f_max=45):

        if self.sub_window_size >= 60:
            segment_window = 10
        else:
            segment_window = 2

        freq, psd, psd_unit = welch_psd(seismic_data_seq,
                                        sampling_freq,
                                        f_min, f_max,
                                        segment_window,
                                        scaling="density",
                                        unit_dB=True)
        f_skew = skew(psd)

        return f_skew


    def prepare_matrix(self, st, print_reminder=True, matrix_dimension=2):
        '''

        Args:
            st: Obspy stream,
            print_reminder: bool,
            matrix_dimension: int, number of measure matrix

        Returns:

        '''

        tr = self.trim_st(st)
        if print_reminder is True:
            print("This action <prepare_matrix> will take while, be patient.")

        st_data = tr.data
        st_startime_array = tr.stats.starttime.timestamp
        st_end_time = tr.stats.endtime.timestamp
        st_sps = tr.stats.sampling_rate
        npts = tr.stats.npts

        chunk_t_str, chunk_t, chunk_x = chunk_data(st_data,
                                                   self.sub_window_size,
                                                   self.window_overlap,
                                                   st_startime_array,
                                                   st_end_time,
                                                   st_sps,
                                                   npts)

        output_matrix = np.empty((chunk_t_str.size, matrix_dimension+2), dtype=object)
        for idx, (t_str, t_float, data) in tqdm(enumerate(zip(chunk_t_str, chunk_t, chunk_x)),
                                                total=len(chunk_t_str),
                                                desc="Progress of <prepare_matrix>",
                                                file=sys.stdout):

            time_array = np.array([t_str, t_float])

            t_snr = self.snr(seismic_data_seq=data)
            f_skew = self.skewness(seismic_data_seq=data, sampling_freq=st_sps)

            output_matrix[idx, :] = np.concatenate((time_array, np.array([t_snr, f_skew])), axis=0)

        return output_matrix
