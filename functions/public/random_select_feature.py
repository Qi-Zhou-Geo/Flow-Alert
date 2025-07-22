#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2025-02-06
#__author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
#__find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission

import random

def generate_random_selected_feature_id(repeate, num_selected, num_total_feature=80):
      '''
      Random select some seismic features for training.

      Args:
            repeate: int, this is make t
            num_selected: int, 1 <= num_selected <= num_total_feature
            num_total_feature: int, number of (Type A + Type B) is 80

      Returns:
            selected_column: List[int], random selected feature
      '''

      random.seed(200 + repeate)

      selected_column = random.sample(range(num_total_feature), num_selected)
      selected_column.sort()

      return selected_column
