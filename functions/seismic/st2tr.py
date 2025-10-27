#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2025-01-20
# __author__ = Qi Zhou and Sibashish Dash, GFZ Helmholtz Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission

import warnings
from obspy import Stream, Trace


def stream_to_trace(st):

    if isinstance(st, Stream):

        if len(st) == 0:
            raise ValueError("Stream is empty, cannot convert to Trace")
        elif len(st) > 1:
            warnings.warn(f"Stream contains {len(st)} traces, returning only the first one")
            tr = st[0]
        else:
            tr = st[0]

    elif isinstance(st, Trace):
        tr = st

    else:
        raise TypeError(f"Expected Stream or Trace, got {type(st).__name__}")

    return tr