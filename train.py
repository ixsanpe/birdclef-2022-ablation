"""
Train a pipeline 
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

DATA_PATH = os.getcwd() + '/birdclef-2022/'
OUTPUT_DIR = 'output/'
LOCAL_TEST = True
WANDB = False

def validate(
    model: nn.Module, 
    data_pipeline_val : nn.Module, 
    val_loader : DataLoader, 
    device: str, 
    criterion: Callable, 
    metric = None,
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
            d_v = data_pipeline_val(d_v)
            x_v, y_v = d_v['x'], d_v['y'].float()
            y_v_logits = model(x_v)
            y_v_pred = torch.sigmoid(y_v_logits)
            y_val_true.append(y_v)
            y_val_pred.append(y_v_pred)
            # running_val_loss += criterion(y_v_logits, y_v)
            running_val_loss += criterion(y_v_pred, y_v)
        
        y_val_true = torch.cat(y_val_true).to('cpu')
        y_val_pred = torch.cat(y_val_pred).to('cpu')
        
        val_loss = running_val_loss/len(val_loader)

        if metric != None:
            val_score = metric(y_val_pred, y_val_true.int())
        else:
            val_score= 0.

    return val_loss, val_score

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
    model: nn.Module, 
    data_pipeline_train: nn.Module, 
    data_pipeline_val: nn.Module, 
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    optimizer: torch.optim.Optimizer, 
    criterion: Callable, 
    metric: Callable, 
    model_saver: callable=None,
    epochs: int=1,
    print_every: int=-1, 
    device: str='cpu',
    name: str="",
    n_splits = 5, 
    log_spectrogram:bool=True, 
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

    """
    Note that the train function is a mess right now, since we are both training and logging at the same time.

    We are refactoring to have a class Trainer that maintains a Logger to resolve this issue and reduce the
    amount of repeated code, but at the moment it is a work in progress. 
    """

    def step(model, d, optimizer, criterion, running_train_loss):
        d = data_pipeline_train(d)
        x, y = d['x'], d['y'].float()
        logits = model(x)
        y_pred = torch.sigmoid(logits)
        optimizer.zero_grad()
        loss = criterion(y_pred, y)
        running_train_loss = running_train_loss + loss.item()
        loss.backward()
        optimizer.step()
        return running_train_loss

    print(f'{20*"#"}\nStarting Training on {device} \n{20*"#"}')

    if WANDB:
        if log_spectrogram:
            wandb_spec_table = wandb.Table(columns=['Sprectrogram', 'Predicted', 'Expected'])

    train_loss, val_loss, val_metric = [], [], []
    for epoch in range(epochs):
        print(f'starting epoch {epoch+1} / {epochs}')

        running_train_loss = 0.
        epoch_train_loss = 0.
        epoch_train_metric = 0. # not used at the moment
        epoch_val_loss = 0.
        epoch_val_metric = 0.

        for i, d in enumerate(train_loader):
            
            # optimization step
            running_train_loss = step(model, d, optimizer, criterion, running_train_loss)
            
            # reporting etc
            if i % print_every == print_every-1: 
                epoch_val_loss, epoch_val_metric = validate(
                    model, 
                    data_pipeline_val, 
                    val_loader, 
                    device, 
                    criterion, 
                    metric
                )
                print_output(
                    running_train_loss/i, 
                    epoch_train_metric, 
                    epoch_val_loss, 
                    epoch_val_metric, 
                    i, 
                    len(train_loader), 
                    epoch
                )


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
             
        if model_saver != None:
            model_saver.save_best_model(epoch_val_loss, epoch, model, optimizer, criterion)

        # Track losses
        train_loss.append(epoch_train_loss)        
        val_loss.append(epoch_val_loss)
        val_metric.append(epoch_val_metric)

    # At the end of training: Save model and training curve
    model_saver.save_final_model(epochs, model, optimizer, criterion) 
    model_saver.save_plots(train_loss, val_loss)
     

def main():
    print('Starting the training...')
    experiment_name = "baseline_" + str(int(time.time())) if not LOCAL_TEST else "local"
    # for pre-processing
    # splitting
    duration = 30 
    n_splits = 6
    test_split = 0.05

    # some hyperparameters
    bs = 2 # batch size
    epochs = 300
    learning_rate = 1e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    N = 200

    if N != -1:
        warnings.warn(f'\n\nWarning! Using only {N} training examples!\n')

    # Load data 
    with open(f'{DATA_PATH}all_birds.json') as f:
        birds = json.load(f)

    num_classes = len(birds)

    metadata = pd.read_csv(f'{DATA_PATH}train_metadata.csv')[:N]

    # train test split
    tts = metadata.sample(frac=test_split).index 
    df_val = metadata.iloc[tts]
    df_train = metadata.drop(tts)

    # Datasets, DataLoaders
    train_data = SimpleDataset(df_train, DATA_PATH, mode='train', labels=birds)
    val_data = SimpleDataset(df_val, DATA_PATH, mode='train', labels=birds)

    train_loader = DataLoader(
        train_data, 
        batch_size=bs, 
        num_workers=4, 
        collate_fn=lambda d: collate_fn(d, device), 
        shuffle=True, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_data, 
        batch_size=bs, 
        num_workers=4, 
        collate_fn=lambda d: collate_fn(d, device), 
        shuffle=False, 
        pin_memory=True
    )

    # create model
    transforms1 = TransformApplier(
        [ 
            SelectSplitData(duration, n_splits, offset=None), 
            # add more transforms here
        ]
    )

    wav2spec = Wav2Spec()

    transforms2 = TransformApplier(
        [
            InstanceNorm()
        ]
    ) 

    data_pipeline_train = nn.Sequential(
        transforms1, 
        wav2spec,
        transforms2, 
    ).to(device)

    data_pipeline_val = nn.Sequential(
        transforms1, 
        wav2spec, 
        transforms2
    ).to(device) 

    # Model Architecture
    cnn = PretrainedModel(
        model_name='efficientnet_b2', 
        in_chans=1, # normally 3 for RGB-images
    )

    # Post-processing
    transforms3 = TransformApplier(
        [
            SimpleAttention(cnn.get_out_dim()), 
            RejoinSplitData(duration, n_splits)
        ]
    )

    output_head = OutputHead(n_in=cnn.get_out_dim() * n_splits, n_out=num_classes)

    # Model definition
    model = nn.Sequential(
        cnn,
        transforms3, 
        output_head,
    ).to(device)

    optimizer = Adam(model.parameters(), lr = learning_rate)
    criterion = nn.BCELoss() # nn.CrossEntropyLoss(weight=None, reduction='mean')

    # Define a scoring metric
    # We are working on this so that we report multiple metrics instead, in order to get
    # a more complete picture of the model performance. 
    # metric = MulticlassF1Score(
    #     num_classes = num_classes, # TODO check this
    #     topk = 1, # this means we say that we take the label with the highest probability for prediction
    #     average='micro' # TODO Discuss that
    # ) 
    metric = Recall( 
        num_classes=num_classes, 
        threshold=.5, 
    ) # Gives a better idea since most predictions are 0 anyways?
    
    model_saver = ModelSaver(OUTPUT_DIR, experiment_name)

    

    config = {
        "epochs": epochs,
        "batch_size": bs,
        "learning_rate": learning_rate,
        "device": device,
        "duration" : duration,
        "n_splits" : n_splits,
        "transforms1": transforms1,
        "transforms2": transforms2,
        "transforms3": transforms3,
        "model": model,
        "test_split" : test_split
    }

    if WANDB:
        wandb.init(project="Baseline", entity="ai4goodbirdclef", name=experiment_name, config=config)
        wandb.watch(model)

    train(
        model, 
        data_pipeline_train,
        data_pipeline_val,  
        train_loader, 
        val_loader, 
        optimizer, 
        criterion,
        metric,
        model_saver=model_saver,
        epochs=epochs, 
        device=device, 
        print_every=10,
        n_splits=n_splits, 
    )

if __name__ == '__main__':
    main()
