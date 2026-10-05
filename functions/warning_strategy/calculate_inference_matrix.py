#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = Last modified: 2026-09-15T18:25:31
# __author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission

import numpy as np
from obspy import UTCDateTime


def inference_matrix(
    array_temp,
    benchmark_time,
    event_start,
    event_end,
    pro_epsilon=0.5,
    buffer1=0.5,
    buffer2=1,
    fmt="%Y-%m-%dT%H:%M:%S",
):
    """
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
    """

    # convert the time
    dt1 = UTCDateTime(benchmark_time)
    benchmark_time_str = dt1.strftime(fmt)
    benchmark_time_float = float(dt1)
    print(benchmark_time_str)

    dt2 = UTCDateTime(event_start)
    event_start_str = dt2.strftime(fmt)
    event_start_float = float(dt2)

    dt3 = UTCDateTime(event_end)
    event_end_str = dt3.strftime(fmt)
    event_end_float = float(dt3)

    if benchmark_time_float >= event_start_float and benchmark_time_float <= event_end_float:
        pass
    else:
        msg = (
            "Check the time reference.\n",
            f"benchmark_time_str: {benchmark_time_str}\n",
            f"benchmark_time_str: {benchmark_time_str}\n",
            f"event_end_str: {event_end_str}",
        )
        raise ValueError(msg)

    # split the array
    t_target = array_temp[:, 0].astype(float)  # float time
    t_str = array_temp[:, 1]
    pro_mean = array_temp[:, -2].astype(float)

    # find the buffer time region
    s_buffer_f = event_start_float - buffer1 * 3600  # float event start time with buffer
    e_buffer_f = event_end_float + buffer2 * 3600  # float event end time with buffer

    # find the cloest time for the given buffer
    s_time_diff = t_target - s_buffer_f
    id_s = np.argmin(np.abs(s_time_diff))

    e_time_diff = t_target - e_buffer_f
    id_e = np.argmin(np.abs(e_time_diff))

    # check if the id_s, id_e is too far away the real time
    if np.min(np.abs(s_time_diff)) >= 600:  # unit of 600 is second
        print(
            f"Warning!\n"
            f"The event buffer time may contain error,\n"
            f"Reason: the buffered START time {t_str[id_s]} is not within 10 minutes of any target time.\n"
            f"benchmark time={benchmark_time}, start time={event_start}\n"
        )

    if np.min(np.abs(e_time_diff)) >= 600:  # unit of 600 is second
        print(
            f"Warning!\n"
            f"The event buffer time may contain error,\n"
            f"Reason: the buffered END time {t_str[id_e]} is not within 10 minutes of any target time.\n"
            f"benchmark time={benchmark_time}, event end={event_end}\n"
        )

    # check the first detection
    index = np.argwhere(pro_mean[id_s:id_e] >= pro_epsilon).flatten()

    if len(index) > 0:
        # model reaches the detection threshold
        detection_type = "model_detect_event"
        temp_id = id_s + index[0]
        first_detection = t_target[temp_id]
        print("first_detection", first_detection)
        increased_warning_time = benchmark_time_float - first_detection
        first_detection_str = UTCDateTime(first_detection).strftime(fmt)
    else:
        # check whether the probability increases within the buffer
        min_increase = 0.05  # 5 percentage points above the buffer's starting probability
        index = np.argwhere(pro_mean[id_s:id_e] - pro_mean[id_s] >= min_increase).flatten()

        if len(index) > 0:
            # probability increases but does not reach the detection threshold
            detection_type = "model_see_event"
            temp_id = id_s + index[0]
            first_detection = t_target[temp_id]
            increased_warning_time = benchmark_time_float - first_detection
            first_detection_str = UTCDateTime(first_detection).strftime(fmt)
            print(
                f"Warning!\n"
                f"Probability increased but did not reach {pro_epsilon}.\n"
                f"benchmark_time = {benchmark_time},\n"
                f"first increase time = {t_str[temp_id]},\n"
                f"buffer time = {t_str[id_s]} to {t_str[id_e - 1]},\n"
                f"probability increased from {pro_mean[id_s]:.5f} "
                f"to {pro_mean[temp_id]:.5f}.\n"
            )

        else:
            # probability does not show a meaningful increase
            detection_type = "failed"
            first_detection = "None"
            increased_warning_time = "None"
            first_detection_str = "1990-01-01T12:00:00"  # this is used to keep the format
            print(
                f"Warning!\n"
                f"Flow-Alert did not detect a meaningful probability increase.\n"
                f"benchmark_time = {benchmark_time},\n"
                f"buffer time = {t_str[id_s]} to {t_str[id_e - 1]},\n"
                f"max predicted probability = {np.max(pro_mean[id_s:id_e]):.5f}.\n"
            )

    # check whether false detection
    index = (
        np.argwhere(pro_mean[:id_s] >= pro_epsilon).flatten().tolist()
        + np.argwhere(pro_mean[id_e:] >= pro_epsilon).flatten().tolist()
    )
    false_detection = len(index)
    false_detection_ratio = false_detection / len(pro_mean)

    return (
        detection_type,
        first_detection,
        first_detection_str,
        increased_warning_time,
        false_detection,
        false_detection_ratio,
    )
