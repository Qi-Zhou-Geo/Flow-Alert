#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2025-12-17
# __author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission

import os
import argparse

import yaml

import numpy as np
import pandas as pd

from obspy import UTCDateTime

# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path

current_dir = Path(__file__).resolve().parent
# using ".parent" on "pathlib.Path" object moves one level up the directory hierarchy

project_root = current_dir.parent.parent
import sys

sys.path.append(str(project_root))


# </editor-fold>

# import the custom functions


def clean_time(time_str1, time_str2, buffer=24):
    # time_str1 is earlier than time_str2

    # get the event-middle time
    dt0 = UTCDateTime(time_str2) - UTCDateTime(time_str1)
    event_duration = float(dt0)

    dt0 = UTCDateTime(time_str1) - event_duration / 2
    # clean the minutes and seconds
    dt0_zero = dt0.replace(second=0, microsecond=0)

    # define the time before and after the buffer time
    dt1 = dt0_zero - buffer * 3600
    dt1_iso_str = dt1.strftime("%Y-%m-%dT%H:%M:%S")

    dt2 = dt0_zero + buffer * 3600
    dt2_iso_str = dt2.strftime("%Y-%m-%dT%H:%M:%S")

    return dt1_iso_str, dt2_iso_str
