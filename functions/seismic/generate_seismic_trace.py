#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-02-23
#__author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
#__find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission

import numpy as np

from obspy import read, Trace, Stream
from obspy.core import UTCDateTime # default is UTC+0 time zone


def create_trace(data, start_time, data_sampling_rate, ref_st=False, return_Trace=False):

    '''
    Create Obspy Trace and Stream
    Args:
        data: numpy 1D data-60s array, unit by m/s or other
        start_time: str, format by "%Y-%m-%dT%H:%M:%S"
        data_sampling_rate: int or float, unit by Hz
        ref_st: bool, obspy Trace or Stream object
        return_Trace: bool,

    Returns:

    '''

    trace = Trace(data=data)
    trace.stats.sampling_rate = data_sampling_rate
    trace.stats.starttime = UTCDateTime(start_time)

    if ref_st is True:
        # with reference stream
        if type(ref_st) is Stream:
            ref_st = ref_st[0]
        elif type(ref_st) is Trace:
            pass

        # get the ref information
        trace.stats.network = ref_st.stats.network
        trace.stats.station = ref_st.stats.station
        trace.stats.channel = ref_st.stats.channel
    else:
        pass

    st = Stream([trace])

    if return_Trace is True:
        st = st[0]

    return st
