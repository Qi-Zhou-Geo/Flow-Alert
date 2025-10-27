#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2025-02-17
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission

import numpy as np
from datetime import datetime, timezone, timedelta
from obspy import UTCDateTime


def inference_matrix(array_temp, benchmark_time, event_start, event_end, pro_epsilon=0.5, buffer1=0.5, buffer2=1):
    '''
    Calculation of the evaluation matrix

    Args:
        array_temp: numpy array, [float timestamps, str timestamps, pro1-N, pro_mean, pro_CI]
        benchmark_time: str, benchmark time
        event_start: str, defined event start time
        event_end: str, defined event end time
        pro_epsilon: float, threshold to seperate the event (>= pro_epsilon) or non-event (< pro_epsilon)
        buffer1: str, unit by hour, buffer the event to avoid the false negative or false positive,
                bacesue the event_start and event_end do not 100% correct.
        buffer2: str, unit by hour, buffer the event to avoid the false negative or false positive

    Returns:
        check the return
    '''

    dt1 = UTCDateTime(benchmark_time)
    benchmark_time = dt1.strftime(f"%Y-%m-%dT%H:%M:%S")
    benchmark_time_float = float(dt1)

    dt2 = UTCDateTime(event_start)
    event_start = dt2.strftime(f"%Y-%m-%dT%H:%M:%S")
    event_start_float = float(dt2)

    dt3 = UTCDateTime(event_end)
    event_end = dt3.strftime(f"%Y-%m-%dT%H:%M:%S")
    event_end_float = float(dt3)

    # split the array
    t_target = array_temp[:, 0].astype(float)
    t_str = array_temp[:, 1]
    pro_mean = array_temp[:, -2].astype(float)

    # find the buffer time region
    dt1 = event_start_float - buffer1 * 3600
    temp_t1 = np.empty((array_temp.shape[0]))
    dt2 = event_end_float + buffer2 * 3600
    temp_t2 = np.empty((array_temp.shape[0]))

    for idx, ts in enumerate(t_target):
        time_diff = ts - dt1 # unit by second
        temp_t1[idx] = np.abs(time_diff)

        time_diff = ts - dt2 # unit by second
        temp_t2[idx] = np.abs(time_diff)

    id_s = np.argmin(temp_t1)
    if np.min(temp_t1) >= 600: # unit of 600 is second
        print(f"Warning!\n"
              f"The event buffer time may contain error,\n"
              f"Reason: the buffered START time {t_str[id_s]} is not within 10 minutes of any target time.\n"
              f"benchmark time={benchmark_time}, start time={event_start}\n")

    id_e = np.argmin(temp_t2)
    if np.min(temp_t2) >= 600: # unit of 600 is second
        print(f"Warning!\n"
              f"The event buffer time may contain error,\n"
              f"Reason: the buffered END time {t_str[id_e]} is not within 10 minutes of any target time.\n"
              f"benchmark time={benchmark_time}, event end={event_end}\n")

    # check the first detection
    index = np.argwhere(pro_mean[id_s:id_e] >= pro_epsilon).flatten()

    if len(index) > 0:
        # default model works
        detection_type = "model_detect_event"
        temp_id = id_s + index[0]
        first_detection = t_target[temp_id]
        increased_warning_time = benchmark_time_float - first_detection
    else:
        # default model does not work
        # check whether the model "see" the patterns
        detection_type = "model_see_event"
        index = np.argmax(pro_mean[id_s:id_e])
        temp_id = id_s + index
        first_detection = t_target[temp_id]
        increased_warning_time = benchmark_time_float - first_detection

        ratio = pro_mean[temp_id] / pro_mean[id_s]
        print(f"Warning!\n"
              f"Flow-Alert failed to detect this event. \n"
              f"benchmark_time = {benchmark_time}, \n"
              f"first_detection = {t_str[temp_id]}, \n"
              f"buffer time = {t_str[id_s]} to {t_str[id_e]}), \n"
              f"max predicted Pro. is {pro_mean[temp_id]} in buffer period, \n"
              f"this is {ratio:.5f} bigger than event start ({t_str[id_s]} -> {pro_mean[id_s]}). \n")

    first_detection_str = UTCDateTime(first_detection).strftime("%Y-%m-%dT%H:%M:%S")

    # check whether false detection
    index = (np.argwhere(pro_mean[:id_s] >= pro_epsilon).flatten().tolist() +
             np.argwhere(pro_mean[id_e:] >= pro_epsilon).flatten().tolist())
    false_detection = len(index)
    false_detection_ratio = false_detection / len(pro_mean)

    return (detection_type, first_detection, first_detection_str, increased_warning_time, false_detection, false_detection_ratio)
