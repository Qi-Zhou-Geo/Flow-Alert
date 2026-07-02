#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = Last modified: 2026-06-19T17:13:09
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission

import obspy
from obspy import UTCDateTime

import scipy
import numpy as np
import pandas as pd
import xarray as xr

print("This is a package-version checker.")
print(f"UTC+0 Time: {UTCDateTime.now().isoformat()}")


print("ObsPy version:", obspy.__version__)
print("SciPy version:", scipy.__version__)
print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)
print("Xarray version:", xr.__version__)

print(f"\n")