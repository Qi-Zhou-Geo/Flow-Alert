#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2025-06-19
#__author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission

import pytz
from datetime import datetime, timedelta

import numpy as np
from obspy import UTCDateTime

def chunk_data(st_data, sub_window_size, window_overlap, st_startime_float, st_endtime_float, st_sps, npts):
    '''
    Chunk the seismic stream to sequence

    Args:
        st_data:
        sub_window_size: int or float, unit by second
        window_overlap: float value in [0, 1], 0 means no overlap
        st_startime_float:
        st_endtime_float:
        st_sps:
        npts:

    Returns:

    '''

    chunk_length = int(st_sps * sub_window_size)  # unit by data point
    step_size = int(chunk_length - chunk_length * window_overlap)  # unit by data point

    # chunk the data
    chunk_x = np.lib.stride_tricks.sliding_window_view(st_data, window_shape=chunk_length)
    chunk_x = chunk_x[::step_size]

    chunk_t = np.linspace(st_startime_float, st_endtime_float, npts)
    chunk_t = np.lib.stride_tricks.sliding_window_view(chunk_t, window_shape=chunk_length)
    chunk_t = chunk_t[::step_size]
    chunk_t = chunk_t[:, 0].reshape(-1)
    chunk_t = chunk_t.astype(float)

    chunk_t_str = [UTCDateTime(i).isoformat() for i in chunk_t]
    chunk_t_str = np.array(chunk_t_str)

    return chunk_t_str, chunk_t, chunk_x

