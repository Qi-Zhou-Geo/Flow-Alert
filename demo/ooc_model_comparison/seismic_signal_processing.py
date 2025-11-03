#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2025-10-14
# __author__ = Kshitij Kar, GFZ Helmholtz Centre for Geosciences
# __find me__ = kshitij.kar@gfz.de, kshitij787.ak@gmail.com, https://github.com/Kshitij301199
# Please do not distribute this code without the author's permission

import os
import sys
import yaml

# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path
current_dir = Path(__file__).resolve().parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys
sys.path.append(str(project_root))
# </editor-fold>

from obspy import UTCDateTime, read, read_inventory, Stream
from typing import List, Literal

def load_seismic_signal(
    continent: List[Literal['Asian', 'African', 'European', 'South_American', 'North_American', 'Oceania']],
    region: str,
    year: int,
    network: str,
    sta: str,
    component: str,
    julian_day: int,
    f_min=1,
    f_max=45,
    ) -> Stream:
    with open(f"{project_root}/config/catchment_code.yaml", "r") as f:
        path = yaml.safe_load(f)
    glic_path = path['glic_sac_dir']
    data_path = f"{glic_path}/{continent}/{region}/{year}/{sta}/{component}"
    pre_data_name = f"{network}.{sta}.{component}.{year}.{str(julian_day-1).zfill(3)}.mseed"
    data_name = f"{network}.{sta}.{component}.{year}.{str(julian_day).zfill(3)}.mseed"
    post_data_name = f"{network}.{sta}.{component}.{year}.{str(julian_day+1).zfill(3)}.mseed"
    meta_data_path = f"{glic_path}/{continent}/{region}/meta_data"
    meta_data_file = [f for f in os.listdir(f"{meta_data_path}") if f.startswith(network)][0]
    print(f"Data File: {data_path}/{data_name}")
    print(f"Meta data file: {meta_data_path}/{meta_data_file}")

    st = Stream()
    try:
        st += read(f"{data_path}/{pre_data_name}")
    except FileNotFoundError:
        pass
    st += read(f"{data_path}/{data_name}")
    try:
        st += read(f"{data_path}/{post_data_name}")
    except FileNotFoundError:
        pass
    st.merge(method=1, fill_value='latest', interpolation_samples=0)
    st._cleanup()
    st.detrend('linear')
    st.detrend('demean')
    st.taper(max_percentage=0.05) # to avoid the "edge" effect in both sides

    inv = read_inventory(f"{meta_data_path}/{meta_data_file}")
    st.remove_response(inventory=inv)
    st.filter("bandpass", freqmin=f_min, freqmax=f_max)
    st.detrend('linear')
    st.detrend('demean')
    st.trim(starttime=UTCDateTime(year=year, julday=julian_day), endtime=UTCDateTime(year=year, julday=julian_day+1))
    st = Stream(st)
    return st
    