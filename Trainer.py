"""
WORK IN PROGRESS!!

The train function in train.py is becoming very messy. This file defines some classes in an attempt to 
resolve the issues that arise from both training and logging at the same time. 

"""


from modules.PretrainedModel import * 
from modules.Wav2Spec import * 
from modules.TransformApplier import * 
from modules.SimpleDataset import * 
from modules.SimpleAttention import * 
from modules.SelectSplitData import *
from modules.Normalization import *
from modules.model_utils import *
from train_utils import ModelSaver, collate_fn

import torch.nn as nn
import pandas as pd 
import json
from torch.utils.data import DataLoader 
from torch.optim import Adam 
from torchvision.utils import make_grid
from typing import Callable
from torchmetrics.classification import MulticlassF1Score, Recall
import wandb
import time
import warnings
import os 

class Logger(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
class EpochLogger():
    def __init__(self, epoch) -> None:
        self.epoch = epoch 
        self.prunning_train_loss = 0.
        self.epoch_train_loss = 0.
        self.epoch_train_metric = 0. # not used at the moment
        self.epoch_val_loss = 0.
        self.epoch_val_metric = 0.

class Trainer(nn.Module):
    def __init__(
        self, 
        model, 
        data_pipeline_train, 
        data_pipeline_val,  
        logger,
        model_saver, 
        criterion, 
        optimizer, 
        device, 
    ):
        super().__init__()
        self.model = model
        self.data_pipeline_train = data_pipeline_train
        self.data_pipeline_val = data_pipeline_val
        self.logger = logger
        self.model_saver = model_saver
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
    


    def validate(
        self, 
        epoch_logger: EpochLogger, 
        val_loader: DataLoader
    ):
        """
        print training progress of the model in terms of training and validation loss
        Parameters:
            model:
                model for which to report progress
            data_pipeline_val: 
                Pre-processing pipeline of the validation data from audio signal to model input
                data_pipeline_val should take a tuple (x, y) as an input
            val_loader: 
                data loader for validation data
            device: 
                device on which to train model
            criterion: 
                criterion (loss) by which to train the model
            running_loss: 
                total accumulated loss over training for this epoch
            epoch:
                current epoch for which to report progress
        """
        with torch.no_grad():
            y_val_true = []
            y_val_pred = []
            running_val_loss = 0.
            for d_v in val_loader:
                d_v = self.data_pipeline_val(d_v)
                x_v, y_v = d_v['x'], d_v['y'].float()
                y_v_logits = self.model(x_v)
                y_v_pred = torch.sigmoid(y_v_logits)
                y_val_true.append(y_v)
                y_val_pred.append(y_v_pred)
                # running_val_loss += criterion(y_v_logits, y_v)
                running_val_loss += self.criterion(y_v_pred, y_v)
            
            y_val_true = torch.cat(y_val_true).to('cpu')
            y_val_pred = torch.cat(y_val_pred).to('cpu')
            
            val_loss = running_val_loss/len(val_loader)

            if self.metric != None:
                val_score = self.metric(y_val_pred, y_val_true.int())
            else:
                val_score= 0.

        epoch_logger.log()

    def print_output(
        train_loss :float = 0., 
        train_metric :float = 0., 
        val_loss :float = 0., 
        val_metric :float=0.,
        i: int=1,
        max_i: int=1,
        epoch :int = 0
    ):
        print(
            f'epoch {epoch+1}, \
            iteration {i}/{max_i}:\t\
            running loss = {train_loss:.3f}\t\
            validation loss = {val_loss:.3f}\t\
            train metric = {train_metric:.3f}\t\
            validation metric = {val_metric:.3f}'
        ) 

    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        epochs: int=1,
        print_every: int=-1, 
    ):
        """
        Train the model 
        Parameters:
            model:
                The model (or model pipeline) to be trained
            data_pipeline_train:
                Pre-processing pipeline of the training data from audio signal to model input
                data_pipeline_train should take a tuple (x, y) as an input
            data_pipeline_val:
                Pre-processing pipeline of the validation data from audio signal to model input
                data_pipeline_val should take a tuple (x, y) as an input
            train_loader:
                data loader for training data
            val_loader:
                data loader for validation data
            optimizer:
                optimizer to train the model
            criterion:
                criterion (loss) by which to train the model
            metric:
                a metric for the validation
            epochs:
                number of epochs for which to train
            print_every:
                how often to report progress. If print_every==-1, we print at the end of every epoch.
            device:
                device on which to train
            n_splits:
                the number of splits of the 30s recording. Essential becasue y_true has shape (B,C), 
                whereas y_pred and x have shape (B*n_splits, C) and (B*n_splits, X, Y)
        
        """

        def step(model, d, optimizer, criterion, running_train_loss):
            d = self.data_pipeline_train(d)
            x, y = d['x'], d['y'].float()
            logits = model(x)
            y_pred = torch.sigmoid(logits)
            optimizer.zero_grad()
            loss = criterion(y_pred, y)
            running_train_loss = running_train_loss + loss.item()
            loss.backward()
            optimizer.step()
            return running_train_loss

        print(f'{20*"#"}\nStarting Training on {self.device} \n{20*"#"}')


        train_loss, val_loss, val_metric = [], [], []
        for epoch in range(epochs):
            print(f'starting epoch {epoch+1} / {epochs}')

            epoch_logger = EpochLogger(epoch=epoch)

            for i, d in enumerate(train_loader):
                
                # optimization step
                running_train_loss = step(self.model, d, self.optimizer, self.criterion, running_train_loss)
                
                # reporting etc
                if i % print_every == print_every-1: 
                    self.validate(epoch_logger, val_loader)
                    # print_output(
                    #     running_train_loss/i, 
                    #     epoch_train_metric, 
                    #     epoch_val_loss, 
                    #     epoch_val_metric, 
                    #     i, 
                    #     len(train_loader), 
                    #     epoch
                    # )


            epoch_train_loss = running_train_loss / len(train_loader)

            if print_every == -1:
                epoch_val_loss, epoch_val_metric = validate(
                    model, 
                    data_pipeline_val, 
                    val_loader, 
                    device, 
                    criterion, 
                    metric
                )
                print_output(
                    epoch_train_loss, 
                    epoch_train_metric, 
                    epoch_val_loss, 
                    epoch_val_metric, 
                    len(train_loader), 
                    len(train_loader), 
                    epoch
                )

            stats = {'train_loss': epoch_train_loss, 'val_loss': epoch_val_loss, 'metric': epoch_val_metric}
            # wandb_log_stats(epoch_train_loss, epoch_val_loss, epoch_val_metric)
            if WANDB:
                wandb.log(stats)
                if log_spectrogram:
                    wandb_log_spectrogram(model, data_pipeline_train, val_loader, device, wandb_spec_table, n_splits)
                
            if self.model_saver != None:
                self.model_saver.save_best_model(epoch_val_loss, epoch, self.model, self.optimizer, self.criterion)

            # Track losses
            train_loss.append(epoch_train_loss)        
            val_loss.append(epoch_val_loss)
            val_metric.append(epoch_val_metric)

        # At the end of training: Save model and training curve
        self.model_saver.save_final_model(epochs, self.model, self.optimizer, self.criterion) 
        self.model_saver.save_plots(train_loss, val_loss)
        