#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2025-08-12
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission

import os
import shutil

# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path
current_dir = Path(__file__).resolve().parent
# using ".parent" on "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys
sys.path.append(str(project_root))
# </editor-fold>


def clean_project(project_root, file_name=".DS_Store", folder_name="__pycache__", size_limit_mb=23):

    '''
    Clean project the project before publish it.

    Args:
        project_root: str, path of the folder
        file_name: str, shit of Mac
        folder_name: str, shit of Python
        size_limit_mb: float or int, maximum file size of single file, make sure Github can host the file

    Returns:
        None

    '''

    for dirpath, dirnames, filenames in os.walk(project_root):
        for file in filenames:
            file_path = os.path.join(dirpath, file)

            # check the file size
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if size_mb > size_limit_mb:
                print(f"Large file (> {size_limit_mb} MB): {file_path} ({size_mb:.2f} MB)")


            # delete the file
            if file == file_name:
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

        # delete the folder
        for dirname in dirnames:
            if dirname == folder_name:
                dir_path = os.path.join(dirpath, dirname)
                try:
                    shutil.rmtree(dir_path)
                    print(f"Deleted directory: {dir_path}")
                except Exception as e:
                    print(f"Error deleting directory {dir_path}: {e}")

clean_project(project_root="/Users/qizhou/Downloads/Flow-Alert-v1.1")
