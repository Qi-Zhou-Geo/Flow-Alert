#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2025-08-12
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission

import os
import shutil


def copy_calculated_feature_files(src_root, dst_root, patterns=("all_A.txt", "all_B.txt", "all_network.txt")):

    for dirpath, dirnames, filenames in os.walk(src_root):
        for file in filenames:
            if file.endswith(patterns):
                src_file = os.path.join(dirpath, file)

                # Recreate relative directory structure
                rel_dir = os.path.relpath(dirpath, src_root)
                dst_dir = os.path.join(dst_root, rel_dir)
                os.makedirs(dst_dir, exist_ok=True)

                dst_file = os.path.join(dst_dir, file)
                shutil.copy2(src_file, dst_file)
                print(f"Copied: {src_file} -> {dst_file}")


copy_calculated_feature_files(
    src_root="/storage/vast-gfz-hpc-01/home/qizhou/3paper/0seismic_feature/European/Illgraben",
    dst_root="/storage/vast-gfz-hpc-01/home/qizhou/3paper/Flow_Alert/data/seismic_feature"
)