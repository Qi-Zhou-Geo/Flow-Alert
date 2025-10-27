#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2025-12-15
# __author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this functions without the author's permission

import os
import pandas as pd
import numpy as np

from datetime import datetime, timezone

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path

current_dir = Path(__file__).resolve().parent
# using ".parent" on "pathlib.Path" object_typeect moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys

sys.path.append(str(project_root))
# </editor-fold>

# import the custom functions


def plotly_1time_series(df, column_x, column_y):

    pio.templates.default = "plotly_white"
    # pio.templates["plotly_white"].layout.font.size = 7
    pio.renderers.default = "chrome"

    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

    # Panel 1
    fig.add_trace(
        go.Scatter(
            x=df[column_x],  # make sure column names match
            y=df[column_y],
            mode='lines',
            line=dict(color='rgba(218, 46, 46, 0.4)', width=2),
            name=column_y
        ),
        row=1, col=1
    )


    # update layout
    fig.update_layout(
        title=column_y,
        xaxis=dict(tickformat="%Y-%m-%dT%H:%M:%S"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        autosize=True,
        width=2000,
        height=800,  # adjust for 4 stacked panels
    )

    # update axes if needed
    fig.update_xaxes(
        title_text=column_x,
        tickformat="%Y-%m-%dT%H:%M:%S",
        hoverformat="%Y-%m-%dT%H:%M:%S",
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(128,128,128,0.5)",  # grey with 50% alpha
        griddash="dash",
        layer="below traces",
        row=1,
        col=1
    )

    fig.update_yaxes(
        title_text=column_y,
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(128,128,128,0.5)",  # grey with 50% alpha
        griddash="dash",
        layer="below traces",
        row=1,
        col=1
    )

    # show in browser
    fig.show(renderer="browser")


def plotly_2time_series(df1, column_x1, column_y1,
                       df2, column_x2, column_y2):

    pio.templates.default = "plotly_white"
    pio.renderers.default = "browser"

    # two rows, shared x-axis
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05
    )

    # -------- Panel 1 --------
    fig.add_trace(
        go.Scatter(
            x=df1[column_x1],
            y=df1[column_y1],
            mode="lines",
            line=dict(color="rgba(218,46,46,0.6)", width=2),
            name=column_y1
        ),
        row=1, col=1
    )

    # -------- Panel 2 --------
    fig.add_trace(
        go.Scatter(
            x=df2[column_x2],
            y=df2[column_y2],
            mode="lines",
            line=dict(color="rgba(46,134,193,0.6)", width=2),
            name=column_y2
        ),
        row=2, col=1
    )

    # -------- Layout --------
    fig.update_layout(
        autosize=True,
        width=2000,
        height=800,
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    # -------- X axis (shared) --------
    fig.update_xaxes(
        tickformat="%Y-%m-%dT%H:%M:%S",
        hoverformat="%Y-%m-%dT%H:%M:%S",
        showgrid=True,
        gridcolor="rgba(128,128,128,0.5)",
        griddash="dash"
    )

    # -------- Y axes --------
    fig.update_yaxes(
        title_text=column_y1,
        row=1, col=1,
        showgrid=True,
        gridcolor="rgba(128,128,128,0.5)",
        griddash="dash"
    )

    fig.update_yaxes(
        title_text=column_y2,
        row=2, col=1,
        showgrid=True,
        gridcolor="rgba(128,128,128,0.5)",
        griddash="dash"
    )

    fig.show()


def plotly_multi_time_series(list_of_tuples,
                             width=2000,
                             height_per_panel=300,
                             shared_title=None):

    """
    list_of_tuples: list of tuples
        [(df, x_col, y_col), ...]
    """

    pio.templates.default = "plotly_white"
    pio.renderers.default = "browser"

    n = len(list_of_tuples)
    if n == 0:
        raise ValueError("No series provided")

    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03
    )

    for i, (df, x_col, y_col) in enumerate(list_of_tuples, start=1):
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[y_col],
                mode="lines",
                name=y_col
            ),
            row=i,
            col=1
        )

        fig.update_yaxes(
            title_text=y_col,
            row=i,
            col=1,
            showgrid=True,
            gridcolor="rgba(128,128,128,0.5)",
            griddash="dash"
        )


    fig.update_xaxes(
        tickformat="%Y-%m-%dT%H:%M:%S",
        hoverformat="%Y-%m-%dT%H:%M:%S",
        showgrid=True,
        gridcolor="rgba(128,128,128,0.5)",
        griddash="dash"
    )

    fig.update_layout(
        autosize=True,
        width=width,
        height=height_per_panel * n,
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        title=dict(
            text=shared_title,
            x=0.5,
            xanchor="center"
        )
    )

    fig.show()