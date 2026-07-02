#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = Last modified: 2026-06-28T17:29:32
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import paramiko

#region ### add the sys.path to search for custom modules ###
from pathlib import Path
current_file = Path(__file__).resolve()
current_dir = current_file.parent

# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys
sys.path.append(str(project_root))
# endregion

# import the custom functions


def connect_sftp(host, port, username, password=None, private_key_path=None):
    
    transport = paramiko.Transport((host, port))
    
    if private_key_path:
        pkey = paramiko.RSAKey.from_private_key_file(private_key_path)  # or Ed25519Key, etc.
        transport.connect(username=username, pkey=pkey)
    else:
        transport.connect(username=username, password=password)
    
    sftp = paramiko.SFTPClient.from_transport(transport)
    
    return sftp, transport


def list_folder(sftp, remote_file_dir):
    
    try:
        file_name = sftp.listdir(remote_file_dir)
        msg = f"Success: {file_name}"
    except Exception as e:
        file_name = []
        msg = f"Exception: {e}"

    file_name = sorted(file_name)
    
    return msg, file_name


def data_exchange(sftp, purpose, local_file_path, remote_file_dir, remote_file_path):

    # Local >> Remote
    if purpose == "upload":

        # (1) check the folder exist or not
        try:
            sftp.stat(remote_file_dir)   # exists
            msg = f"Folder <remote_file_dir> already exist.\n{remote_file_dir}"
        except IOError:
            sftp.mkdir(remote_file_dir)  # create if missing
            msg = f"Successfully created the folder: {remote_file_dir}"
        
        sftp.chmod(remote_file_dir, mode=0o777)
        
        # (2) try to delete the file first
        try:
            sftp.remove(remote_file_path)
        except IOError as e:
            pass
    
        # (3) upload file, overwrites by default
        sftp.put(local_file_path, remote_file_path)
        
    # Remote >> Local
    elif purpose == "download":
        
        # (1) make local folder, ignore if it already exists
        local_file_path = Path(local_file_path)
        local_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        sftp.get(remote_file_path, local_file_path)
        
        msg = f"Successfully downloaded the data: {remote_file_path}"
        
    else:
        raise ValueError(f"check your purpose: {purpose}")
        
    return msg



