#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2026-05-31
# __author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission


from pathlib import Path
import logging


def setup_logger(output_dir, 
                 log_filename, 
                 level=logging.INFO, 
                 force_reset=True):

    log_path = Path(output_dir) / log_filename
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # delete old log
    if force_reset and log_path.exists():
        log_path.unlink()

    # avoid duplicate handlers if called multiple times
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
