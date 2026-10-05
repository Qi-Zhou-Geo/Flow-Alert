#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = Last modified: 2026-09-14T09:41:40
# __author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this functions without the author's permission

import numpy as np
from obspy import read, Stream


# region ### add the sys.path to search for custom modules ###
import sys
from pathlib import Path


current_file = Path(__file__).resolve()
current_dir = current_file.parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent

sys.path.append(str(project_root))
# endregion


def synthetic_stream(
    # raw obspy stream, unit by counts
    st_raw,
    # for pading
    left_pad=1800,
    left_slice=60,
    right_pad=1800,
    right_slice=60,
    # random seed
    seed=1234,
):
    """
    Pad seismic data with synthetic Gaussian noise.

    Args:
        st_raw (obspy.Stream): Raw seismic stream in counts.

        left_pad (float): Left padding length in seconds.
        left_slice (float): Data length used to estimate left-side noise.

        right_pad (float): Right padding length in seconds.
        right_slice (float): Data length used to estimate right-side noise.

        seed (int): Random seed for reproducibility.

    Returns:
        st_new (obspy.Stream): Stream with synthetic noise padding.
    """

    st = st_raw.copy()
    st.merge(method=1, fill_value="latest", interpolation_samples=0)

    tr = st[0]
    sps = tr.stats.sampling_rate
    data = tr.data.astype(float)

    # random generator with fixed seed
    rng = np.random.default_rng(seed)

    # left padding
    if left_pad is not None:
        slice_length = int(left_slice * sps)

        # use beginning of trace as reference
        template = data[:slice_length]
        mean = np.mean(template)
        std = np.std(template)

        n_left = int(left_pad * sps)
        left_noise = rng.normal(mean, std, n_left)
        tr.data = np.concatenate([left_noise, data])

        # move start time earlier
        tr.stats.starttime = tr.stats.starttime - left_pad

    # right padding
    if right_pad is not None:
        slice_length = int(right_slice * sps)

        # use end of trace as reference
        template = data[-slice_length:]
        mean = np.mean(template)
        std = np.std(template)

        n_right = int(right_pad * sps)
        right_noise = rng.normal(mean, std, n_right)
        tr.data = np.concatenate([tr.data, right_noise])

    st_new = Stream(tr)

    return st_new


def usage():

    # load the seismic data
    print(
        "Please run this to get the data: https://github.com/Qi-Zhou-Geo/Flow-Bench/blob/geo/pipeline/Kodari/fetch_data.py"
    )
    st_path = Path("/Users/qizhou/#python/Flow-Alert/demo/Kodari/st_raw.mseed")
    st = read(st_path)

    st.plot()
    print(st[0].stats)

    # pad the data
    st_new = synthetic_stream(st_raw=st)
    st_new.plot()
    print(st_new[0].stats)  # type: ignore
