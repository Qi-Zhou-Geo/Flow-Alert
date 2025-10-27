#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-02-23
#__author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
#__find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission

import os
import logging

from typing import List

from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from datetime import datetime


# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path
current_dir = Path(__file__).resolve().parent
# using ".parent" on "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys
sys.path.append(str(project_root))
# </editor-fold>

# import the custom functions
#from functions.model.results_visualization import *
from functions.data_process.archive_data import *

def _save_checkpoint(model, output_dir, output_format, epoch):

    ckp = model.state_dict()
    path = f"{output_dir}/{output_format}.pt"
    torch.save(ckp, path)

    print(f"Model saved at epoch {epoch}, {path}", flush=True)

def _plot(tensor_temp, purpose, output_dir, output_format):

    if purpose != "training":
        # probability map
        try:
            visualize_probability_map(tensor_temp[:, 0],
                                      tensor_temp[:, 1],
                                      tensor_temp[:, 3],
                                      purpose,
                                      output_dir,
                                      output_format)
        except Exception as e:
            print(f"{purpose}, {output_format}, {e} in visualize_probability_map")
    else:
        pass

    # confusion matrix
    try:
        visualize_confusion_matrix(tensor_temp[:, 2],
                                   tensor_temp[:, 4],
                                   purpose,
                                   output_dir,
                                   output_format)
    except Exception as e:
        print(f"{purpose}, {output_format}, {e} in visualize_confusion_matrix")


def prepare_records(t_target, target, raw_logits, DF_threshold, temperature_scaling, purpose):

    # predicted probability of debris flow label 1
    obs_y_label = (target >= DF_threshold).float()

    if purpose == "training":
        # no temperature scaling
        temp = raw_logits
    else:
        # set temperature scaling
        temp = raw_logits / temperature_scaling

    predicted_pro = torch.softmax(temp, dim=1)[:, 1]
    predicted_pro = torch.round(predicted_pro * 1000) / 1000

    # pre_y_label is for logging/visualization only, not used in training
    pre_y_label = (predicted_pro >= DF_threshold).float()

    # logout logits for checking
    logit_0, logit_1 = raw_logits[:, 0], raw_logits[:, 1]

    record = torch.cat(
        (
            t_target.view(-1, 1),
            target.view(-1, 1),
            obs_y_label.view(-1, 1),
            predicted_pro.view(-1, 1),
            pre_y_label.view(-1, 1),
            logit_0.view(-1, 1),
            logit_1.view(-1, 1),
        ),
        dim=1)  # as column

    return record


class Train_Test:

    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler._LRScheduler,

                 train_dataloader: List[DataLoader],
                 test_dataloader: List[DataLoader],

                 device: str,

                 # input_format
                 output_dir: str = None,
                 input_format: str = "9S-2017_2019-ILL12-EHZ-C",
                 model_type: str = "results",

                 # Non-DF label : DF label = 1-class_weight : class_weight
                 class_weight: float = 0.9,  # weight for DF label

                 data_type: str = "feature",

                 DF_threshold: float = 0.5, # threshold to seperate the DF and Non-DF
                 ) -> None:

        # <editor-fold desc="set parameter that used in class">
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.device = device
        self.max_norm_clip = 1
        self.DF_threshold = DF_threshold
        # initial guess
        self.temperature_scaling = torch.nn.Parameter(torch.tensor(1.5)).to(device)
        # </editor-fold>


        # <editor-fold desc="set class output dir and format">
        if output_dir is None:
            current_dir = Path(__file__).resolve().parent
            project_root = current_dir.parent.parent
            output_dir = f"{project_root}/output"

        self.output_dir = f"{output_dir}/{model_type}"
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_format = input_format
        # </editor-fold>


        # <editor-fold desc="set initial monitor parameter value">
        # be carfule the nagative symbol "-"
        self.train_monitor = 1e3 # monitor epoch loss
        self.test_monitor = -1e3 # monitor F1
        self.validate_monitor = -1e3 # monitor F1
        # </editor-fold>


        # <editor-fold desc="set loss func">
        if data_type == "feature":
            class_weight_binary = torch.tensor([1-class_weight, class_weight]).to(self.device)
        elif data_type == "waveform":
            class_weight_binary = torch.tensor([1-class_weight, class_weight]).to(self.device)# [0.004, 0.996]
        else:
            print(f"check the training/testing data type, data_type={data_type}")
        self.loss_func = torch.nn.CrossEntropyLoss(reduction="mean", weight=class_weight_binary)
        # </editor-fold>

    def training(self, epoch, dump_model=False, purpose="training"):
        self.model.train()
        tensor_temp = torch.empty((0, 7)).to(self.device)
        epoch_loss, dataloader_leangth = 0, 0

        # first loop all dataloader
        for dataloader in self.train_dataloader:
            # second loop all batch
            dataloader_leangth += len(dataloader)
            for batch_data in dataloader:
                # t_features of t to t_{sequence_length}, shape ([batch_size, sequence_length]), float time stamps
                t_features = batch_data['t_features'].to(self.device)
                # features of t to t_i, shape ([batch_size, sequence_length, num_stations * num_channels])
                features = batch_data['features'].to(self.device)

                # t_target of t_{sequence_length+1}, shape ([batch_size, 1]), float time stamps
                t_target = batch_data['t_target'].to(self.device)
                # target of t_{sequence_length+1}, shape ([batch_size, 1]), debris flow probability or label
                target = batch_data['target'].to(self.device)

                raw_logits = self.model(features, t_features)  # return the model output logits, shape (batch_size, 2)
                loss = self.loss_func(raw_logits, target)
                epoch_loss += loss.item()

                # update the gredient
                self.optimizer.zero_grad()
                loss.backward()
                # clip the grad to avoid the explosion
                total_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.max_norm_clip,
                    norm_type=2
                )
                if total_norm > 10:
                    print(f"Clipped from {total_norm:.2f} to {self.max_norm_clip}")

                self.optimizer.step()

                # predicted probability of debris flow label 1
                record = prepare_records(t_target, target, raw_logits,
                                         DF_threshold=self.DF_threshold,
                                         temperature_scaling=self.temperature_scaling,
                                         purpose=purpose)
                tensor_temp = torch.cat((tensor_temp, record), dim=0)  # as row

        epoch_loss /= dataloader_leangth

        # make sure the first column is time stamps
        tensor_temp = tensor_temp.detach().cpu().numpy()
        tensor_temp = tensor_temp[tensor_temp[:, 0].argsort()]

        # dump the results
        be_saved_array = tensor_temp
        evaluate_matrix = dump_evaluate_matrix(be_saved_array,
                                               f"{purpose}, epoch_{epoch}_{self.output_format}",
                                               f"{self.output_dir}", "summary_LSTM_all",
                                               dump=dump_model)
        # update the training loss
        if epoch_loss < self.train_monitor:
            self.train_monitor = epoch_loss
            # if epoch decrease, dump the training results
            dump_model_prediction(be_saved_array, purpose, self.output_dir, self.output_format)


        if dump_model is True:
            _save_checkpoint(self.model, self.output_dir, f"{purpose}-{self.output_format}-at-{epoch}", epoch)


        return epoch_loss, evaluate_matrix

    def testing(self, epoch, purpose, received_dataloader):
        self.model.eval()
        # loop all testing data-60s
        with torch.no_grad():
            # first loop all dataloader
            for idx, dataloader in enumerate(received_dataloader):
                tensor_temp = torch.empty((0, 7)).to(self.device)
                epoch_loss = 0
                # second loop all batch
                for batch_data in dataloader:
                    # t_features of t to t_{sequence_length}, shape ([batch_size, sequence_length]), float time stamps
                    t_features = batch_data['t_features'].to(self.device)
                    # features of t to t_i, shape ([batch_size, sequence_length, num_stations * num_channels])
                    features = batch_data['features'].to(self.device)

                    # t_target of t_{sequence_length+1}, shape ([batch_size, 1]), float time stamps
                    t_target = batch_data['t_target'].to(self.device)
                    # target of t_{sequence_length+1}, shape ([batch_size, 1]), debris flow probability or label
                    target = batch_data['target'].to(self.device)

                    raw_logits = self.model(features, t_features)  # return the model output logits, shape (batch_size, 2)
                    loss = self.loss_func(raw_logits, target)
                    epoch_loss += loss.item()

                    # predicted probability of debris flow label 1
                    record = prepare_records(t_target, target, raw_logits,
                                             DF_threshold=self.DF_threshold,
                                             temperature_scaling=self.temperature_scaling,
                                             purpose=purpose)
                    tensor_temp = torch.cat((tensor_temp, record), dim=0)  # as row

                epoch_loss /= len(dataloader)

                # make sure the first column is time stamps
                tensor_temp = tensor_temp.detach().cpu().numpy()
                tensor_temp = tensor_temp[tensor_temp[:, 0].argsort()]

                # dump the results
                be_saved_array = tensor_temp
                evaluate_matrix = dump_evaluate_matrix(be_saved_array,
                                                       f"{purpose}, epoch_{epoch}_{self.output_format}",
                                                       f"{self.output_dir}", "summary_LSTM_all")

                if evaluate_matrix[-2] > self.test_monitor:
                    dump_model_prediction(be_saved_array, purpose, self.output_dir, self.output_format)

        return epoch_loss, evaluate_matrix

    def activation(self, num_epoch=50, testing=True):

        # initial as none
        train_matrix_temp = ""
        test_matrix_temp = ""

        for epoch in range(1, num_epoch+1): # loop 50 times for training
            train_loss, train_matrix = self.training(epoch)  # train the model every epoch
            self.scheduler.step(train_loss)

            if testing is True:
                test_loss, test_matrix = self.testing(epoch, "testing", self.test_dataloader)
            else:
                test_loss, test_matrix = -1, [-1] * 8

            # check the test F1
            if test_matrix[-2] > self.test_monitor:
                self.test_monitor = test_matrix[-2] # update the monitor
                train_matrix_temp = f"training, epoch_{epoch}_{self.output_format}, {train_matrix} \n"
                test_matrix_temp = f"testing, epoch_{epoch}_{self.output_format}, {test_matrix} \n"
                _save_checkpoint(self.model, self.output_dir, f"{self.output_format}", epoch)

            # record the logs
            time_now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            record = f"epoch, {epoch}/{num_epoch}, UTC+0, {time_now}, train_loss, {train_loss}, test_loss, {test_loss}, " \
                     f"self.train_monitor, {self.train_monitor}, self.test_monitor, {self.test_monitor}, test_matrix, {test_matrix}\n"
            print(record)

        # dump the optimal matrix to local
        dump_as_row(output_dir=f"{self.output_dir}",
                    output_name="summary_LSTM_optimal",
                    variable_str=train_matrix_temp)
        dump_as_row(output_dir=f"{self.output_dir}",
                    output_name="summary_LSTM_optimal",
                    variable_str=test_matrix_temp)
