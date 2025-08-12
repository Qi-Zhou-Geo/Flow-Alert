#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-02-23
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission

import os
import json
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler


plt.rcParams.update({'font.size': 7})  # , 'font.family': "Arial"})

def delete_png(folder_path, file_prefix):

    for file_name in os.listdir(folder_path):
        if file_name.startswith(file_prefix):
            file = os.path.join(folder_path, file_name)

            try:
                os.remove(file)
            except Exception as e:
                print(f"Error {e}"
                      f"when delete : {file}")


def visualize_feature_imp(imp_source, imp, input_features_name,
                          input_station, model_type, feature_type, input_component, num_feats):

    # <editor-fold desc="add the sys.path to search for custom modules">
    from pathlib import Path
    current_dir = Path(__file__).resolve().parent
    # using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
    project_root = current_dir.parent
    import sys
    sys.path.append(str(project_root))
    # </editor-fold>

    scaler = MinMaxScaler()

    imp = scaler.fit_transform(imp.reshape(-1, 1)).reshape(-1)
    arr = np.column_stack((input_features_name, imp))
    feature_imp_mapping = dict(zip(input_features_name, imp))
    json_file_path = '/home/kshitkar/ML-BL-enhance-DF-EWs/functions/feature_mapping.json'
    with open(json_file_path, 'r') as file:
        feature_mapping = json.load(file)

    folder_path = f"{project_root}/trained_model/figures_{num_feats}"
    os.makedirs(folder_path, exist_ok=True)
    np.savetxt(f"{folder_path}/{input_station}_{model_type}_{feature_type}_{input_component}_{imp_source}_IMP.txt",
               arr, fmt='%s', delimiter=',')

    # Fill in the final_mapping with the normalized importance values for features present in input_features_name
    for feature in feature_mapping:
        if feature in input_features_name:
            feature_mapping[feature] = feature_imp_mapping[feature]
        else:
            feature_mapping[feature] = 0.0  # Set extra features to zero

    fig = plt.figure(figsize=(5.5, 3))
    ax1 = fig.add_subplot(1, 1, 1)
    imp = np.array(list(feature_mapping.values()))
    sns.barplot(x= np.arange(len(feature_mapping)), y= imp, label=feature_type.upper())

    featureIDboundary = np.array([[-0.5, 10], [11, 35], [36, 52], [53, 69], [70, 79.5]])
    for step in range(featureIDboundary.shape[0]):
        if step % 2 == 0:
            facecolor = "black"
        else:
            facecolor = "grey"
        plt.axvspan(xmin=featureIDboundary[step, 0], xmax=featureIDboundary[step, 1],
                    ymin=0, ymax=1, alpha=0.2, edgecolor="None", facecolor=facecolor)

    # plot features name
    arrSort = np.argsort(arr[:, 1])[::-1]
    xID = [12, 22, 32, 42, 52] + [12, 22, 32, 42, 52]
    yID = [np.nanmax(imp) * 0.8, np.nanmax(imp) * 0.8, np.nanmax(imp) * 0.8, np.nanmax(imp) * 0.8, np.nanmax(imp) * 0.8,
           np.nanmax(imp) * 0.7, np.nanmax(imp) * 0.7, np.nanmax(imp) * 0.7, np.nanmax(imp) * 0.7, np.nanmax(imp) * 0.7]
    try:  # some feature_type may do not have 10 features
        for step in range(10):
            plt.text(x=xID[step], y=yID[step], s=f"ID{arrSort[step]}: {arr[arrSort[step], 0]}", fontsize=5)
    except Exception as e:
        print(e)

    plt.xlim(-0.5, 79.5)
    plt.grid(axis='y', ls="--", lw=0.5)
    plt.legend(loc="upper right", fontsize=5)

    plt.xlabel(f"Feature ID, station: {input_station}, IMP_source: {imp_source}", weight='bold')
    plt.ylabel('Features Importance', weight='bold')

    ax1.xaxis.set_major_locator(ticker.MultipleLocator(10))
    plt.tight_layout()
    plt.savefig(f"{folder_path}/{input_station}_{model_type}_{feature_type}_{input_component}_{imp_source}_IMP.png", dpi=600)
    plt.close(fig)


def visualize_confusion_matrix(output_dir, obs_y_label, pre_y_label, training_or_testing,
                               input_station, model_type, feature_type, input_component, num_feats=80):

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # get the parent path
    folder_path = f"{output_dir}/figures_{num_feats}"
    os.makedirs(folder_path, exist_ok=True)
    
    # delete it if this file exist
    delete_png(folder_path=f"{folder_path}/",
               file_prefix=f"{input_station}_{model_type}_{feature_type}_{training_or_testing}_{input_component}")

    cm_raw = confusion_matrix(obs_y_label, pre_y_label)
    cm_df_raw = pd.DataFrame(cm_raw, index=["0:Non-DF", "1:DF"], columns=["0:Non-DF", "1:DF"])

    cm_normalize = confusion_matrix(obs_y_label, pre_y_label, normalize='true')
    cm_df_normalize = pd.DataFrame(cm_normalize, index=["0:Non-DF", "1:DF"], columns=["0:Non-DF", "1:DF"])

    f1 = f1_score(obs_y_label, pre_y_label, average='binary', zero_division=0)

    fig = plt.figure(figsize=(3, 3))
    sns.heatmap(
        cm_df_raw,
        xticklabels=1,
        yticklabels=1,
        annot=True,
        fmt='.0f',
        square=True,
        cmap=sns.color_palette(["white"]),  # makes all cells white
        cbar=False,
        annot_kws={'color': 'black', 'fontsize': 6}
    )

    plt.text(x=0.5, y=0.4, s=f"{cm_df_normalize.iloc[0, 0]:.4f}", ha='center', color="black", fontsize=6)
    plt.text(x=1.5, y=0.4, s=f"{cm_df_normalize.iloc[0, 1]:.4f}", ha='center', color="black", fontsize=6)

    plt.text(x=0.5, y=1.4, s=f"{cm_df_normalize.iloc[1, 0]:.4f}", ha='center', color="black", fontsize=6)
    plt.text(x=1.5, y=1.4, s=f"{cm_df_normalize.iloc[1, 1]:.4f}", ha='center', color="black", fontsize=6)

    plt.ylabel("Actual Class", weight='bold')
    plt.xlabel(f"Predicted Class" + "\n" + f"{training_or_testing}, {input_station}, F1={f1:.4}", weight='bold')

    plt.tight_layout()
    plt.savefig(
        f"{folder_path}/{input_station}_{model_type}_{feature_type}_{training_or_testing}_{input_component}_F1_{f1:.4f}.png",
        dpi=600)

    plt.close(fig)

    return f1
