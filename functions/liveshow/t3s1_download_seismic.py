#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = Last modified: 2026-07-03T23:28:31
# __author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this functions without the author's permission

import os
import logging

from obspy import UTCDateTime, read, Stream, read_inventory

#region ### add the sys.path to search for custom modules ###
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
current_dir = current_file.parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent

sys.path.append(str(project_root))
# endregion


# import the custom functions
from functions.download_WSL.meta_sftp import sftp_metadata
from functions.download_WSL.fetch_sftp import connect_sftp, list_folder, data_exchange


def julday_to_be_checked():
    
    # time now
    t_now = UTCDateTime.now()
    julday = t_now.julday
    hour = t_now.hour
    minute = t_now.minute

    # is current time in the mid-night: t_now <= Year-Month-Day2T00:05:00
    in_range = (hour == 0 and minute <= 5)
    if in_range is True:
        # check yesterday and today
        j_list = [julday - 1, julday]
    else:
        # only check today
        j_list = [julday]

    return j_list


def check_remote_data(logger, j_list, remote_sub_folder):
    
    # build the SFTP connection
    host, port, username, password, private_key_path, remote_dir = sftp_metadata()
    sftp, transport = connect_sftp(host, port, username, password, private_key_path)

    # check remote file
    all_remote_data = []
    for julday in j_list:
        remote_file_dir = Path(remote_dir) / f"{remote_sub_folder}/{str(julday).zfill(3)}"
        remote_file_dir = f"{remote_file_dir}"
        msg, file_name = list_folder(sftp, remote_file_dir)
        
        # replace the previous message
        msg = f"Find: {len(file_name)} files in {remote_file_dir}"
        
        if isinstance(logger, logging.Logger):
            logger.info(msg)
        else:
            print(msg)
        
        file_name = sorted(file_name)
        # format as: 183=9S.ILL12..EHZ_2026-07-02T06-58-37.mseed
        temp = [str(julday) + "=" + x for x in file_name]
        all_remote_data = all_remote_data + temp

    
    # close the sftp
    sftp.close() # type: ignore
    transport.close()
   
    return all_remote_data


def check_local_data(j_list, local_sub_folder):
    
    # check local file
    all_local_data = []
    for julday in j_list:
        local_file_dir = Path(project_root) / f"{local_sub_folder}/{str(julday).zfill(3)}"
        local_file_dir.mkdir(parents=True, exist_ok=True)
        local_file_dir = f"{local_file_dir}"
        
        file_name = os.listdir(local_file_dir)
        file_name = sorted(file_name)
        
        # format as: 183=9S.ILL12..EHZ_2026-07-02T06-58-37.mseed
        temp = [str(julday) + "=" + x for x in file_name]
        all_local_data = all_local_data + temp
        
    return all_local_data


def download_data_to_local(logger, new_data_name, remote_sub_folder, local_sub_folder):
    
    # build the SFTP connection
    host, port, username, password, private_key_path, remote_dir = sftp_metadata()
    sftp, transport = connect_sftp(host, port, username, password, private_key_path)

    for file_name in new_data_name:

        julday, seismic_data = file_name.split("=")
        
        # Local file dir. and file path
        local_file_dir = Path(project_root) / f"{local_sub_folder}/{str(julday).zfill(3)}"
        local_file_dir.mkdir(parents=True, exist_ok=True)
        local_file_dir = f"{local_file_dir}"

        local_file_path = Path(project_root) / f"{local_sub_folder}/{str(julday).zfill(3)}" / seismic_data
        local_file_path = f"{local_file_path}"


        # Remote file dir. and file path
        remote_file_dir = Path(remote_dir) / f"{remote_sub_folder}/{str(julday).zfill(3)}"
        remote_file_dir = f"{remote_file_dir}"

        remote_file_path = Path(remote_dir) / f"{remote_sub_folder}/{str(julday).zfill(3)}" / seismic_data
        remote_file_path = f"{remote_file_path}"


        # usage 2
        purpose = "download"
        msg = data_exchange(sftp, purpose, local_file_path, remote_file_dir, remote_file_path)
        msg = f"{purpose.capitalize()}: {seismic_data}\n{msg}\n"
        
        if isinstance(logger, logging.Logger):
            logger.info(msg)
        else:
            print(msg)

    # close the sftp
    sftp.close() # type: ignore
    transport.close()


def merge_seismic_data(logger, local_sub_folder):
    # time now
    t_now = UTCDateTime.now()
    julday = t_now.julday - 1 # merge yesterday's data
    
    
    local_file_dir = Path(project_root) / f"{local_sub_folder}/{str(julday).zfill(3)}"
    local_file_dir.mkdir(parents=True, exist_ok=True)
    
    all_mseed_file = os.listdir(local_file_dir)
    
    st = Stream()
    for file_name in all_mseed_file:
        try:
            st = st + read(f"{local_file_dir}/{file_name}")
        except Exception as e:
            pass
    
    # merge and save to local
    st.merge(method=1, fill_value='latest', interpolation_samples=0)
    mseed_path = Path(project_root) / f"{local_sub_folder}/9S.ILL12.EHZ.{t_now.year}.{str(julday).zfill(3)}.mseed"
    mseed_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # save to local
        st.write(mseed_path, format='MSEED')
        msg1 = f"Data was saved at: {mseed_path}"
        
        
        stats = st[0].stats # type: ignore
        start_time = stats.starttime
        end_time = stats.endtime

        delta_t = end_time - start_time
        if delta_t < 3600 * 24:
            msg2 = (f"julday={julday}, t_now={t_now}\n"
                    f"Data is less than 24 hours.\n"
                    f"st[0].stats={stats}")
        else:
            msg2 = (f"julday={julday}, t_now={t_now}\n"
                    f"Data is all good.\n"
                    f"st[0].stats={stats}")
        
        msg = f"{msg1}\n{msg2}"
        
    except Exception as e:
        msg = f"Error! {e}"

    if isinstance(logger, logging.Logger):
        logger.info(msg)
    else:
        print(msg)
    
    return msg


def load_st(all_local_data, local_sub_folder, f_min=1.0, f_max=25.0, data_length=2):
    
    st = Stream()
    for file_name in all_local_data:
        
        julday, seismic_data = file_name.split("=")
        
        temp_dir = Path(project_root) / f"{local_sub_folder}/{str(julday).zfill(3)}" / seismic_data
        st = st + read(temp_dir)

    # remove sensor response
    tr = st.copy()
    
    tr.merge(method=1, fill_value='latest', interpolation_samples=0)
    tr._cleanup()
    tr.detrend("linear")
    tr.detrend("demean")
    tr.taper(max_percentage=0.01)
    
    inv_path = Path(project_root) / f"deploy/9S_2026.xml"
    inv = read_inventory(inv_path)
    tr.remove_response(inventory=inv)      
       
    tr.filter("bandpass", freqmin=f_min, freqmax=f_max)
    tr.detrend("linear")
    tr.detrend("demean")
    tr.taper(max_percentage=0.01)
    
    # only return the last 3 hours data
    stats = tr[0].stats  # type: ignore
    end_time = stats.endtime
    start_time = stats.endtime - 3600 * data_length
    
    if end_time - start_time >= 3600 * data_length:
        # with enough data
        tr.trim(starttime=start_time, endtime=end_time)
    else:
        pass
    
    return tr


def data_stream_pipeline(logger, remote_sub_folder="wsl_gfz", local_sub_folder="deploy/liveshow_cache/seismic"):
    
    j_list = julday_to_be_checked()
    all_remote_data = check_remote_data(logger, j_list, remote_sub_folder)
    all_local_data = check_local_data(j_list, local_sub_folder)


    # new file in remote, but not in local
    new_data_name = list(set(all_remote_data) - set(all_local_data))
    
    if len(new_data_name) > 0:
        is_new_data = True
        download_data_to_local(logger, new_data_name, remote_sub_folder, local_sub_folder)
        
        st = load_st(all_local_data, local_sub_folder, f_min=1.0, f_max=25.0)
    else:
        is_new_data = False
        st = None
    
    return is_new_data, st
