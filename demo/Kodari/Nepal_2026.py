#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = Last modified: 2026-08-27T15:35:11
# __author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this functions without the author's permission

from obspy import UTCDateTime, read

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
from functions.model.interface_model import FlowAlert


# load the seismic data
print(
    "Please run this to get the data: https://github.com/Qi-Zhou-Geo/Flow-Bench/blob/geo/pipeline/Kodari/fetch_data.py"
)
st_path = Path("/Users/qizhou/#python/Flow-Alert/demo/Kodari/st_cooked.mseed")
st = read(st_path)

# this is the time around the first event
starttime = UTCDateTime("2026-08-26T02:52:00")  # UTC+0
endtime = UTCDateTime("2026-08-26T02:55:00")
st_copy = st.copy()
st_copy = st_copy.trim(starttime, endtime)
st_copy.plot()


# this is the time period used for testing the Flow-Alert
starttime = UTCDateTime("2026-08-26T01:00:00")  # UTC+0
endtime = UTCDateTime("2026-08-26T06:00:00")
st_copy = st.copy()
st_copy = st_copy.trim(starttime, endtime)
st_copy.plot()


# load the model and run it
sub_window_size = 10  # unit by second
window_overlap = 0.8
model_version = "v1dot3model"
model_type = "LSTM"
output_path = Path(project_root) / "demo/Kodari"

flow_alert = FlowAlert(
    model_type,
    model_version,
    st=st_copy,
    output_path=output_path,
    sub_window_size=sub_window_size,
    window_overlap=window_overlap,  # type: ignore
    clip_anomaly=True,
    num_cpus=1,
)

flow_alert.model_config()
model_list = flow_alert.load_model()

feature_arr = flow_alert.prepare_feature()
model_output = flow_alert.make_prediction(tested_model=model_type)

# plot results
event_start, event_end = None, None  # ref: marked by QZ
benchmark_time = "2026-08-26T02:52:10"  # ref: https://earthquake.usgs.gov/earthquakes/eventpage/us7000tbwb/executive

flow_alert.plot(event_start, event_end, benchmark_time)
