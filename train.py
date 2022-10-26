"""
Train a pipeline 
"""

from modules.PretrainedModel import * 
from modules.OnlyXTransform import *
from modules.Wav2Spec import * 
from modules.TransformApplier import * 
from modules.SimpleDataset import * 
from modules.SimpleAttention import * 
from modules.SelectSplitData import *
from modules.Normalization import *
from utils import ModelSaver

import torch.nn as nn
import pandas as pd 
import json
from torch.utils.data import DataLoader 
from torch.optim import Adam 
from typing import Callable
from torchmetrics.classification import MulticlassF1Score
import wandb
import time
import warnings


DATA_PATH = 'data/'
OUTPUT_DIR = 'output/'

def print_probability_ranking(y, n=5):
    assert n <= len(y)

    y = nn.functional.softmax(y, dim=0)

    output = ""
    sorted, indices = torch.sort(y)

    for i in range(n):
        output += "#%i   Class: %i   Prob: %.3f\n"%(i, indices[i], sorted[i])

    return output

def wandb_log_stats(
    train_loss: list = [], 
    val_loss: list=[], 
    val_metric: list=[]
):
    """
    sends the current training statistic to Weights and Biases
    Parameters:
        train_loss:
            training loss, evaluated by cost function on training set
        val_loss: 
            validation loss, evaluated by cost function on validation set
        val_metric: 
            validation metric, evaluated by metric function on validation set
    """

    wandb.log({"train_loss": train_loss,
        "val_loss": val_loss,
        "val_metric" : val_metric})


def wandb_log_spectrogram(
    model,
    data_pipeline_val,
    val_loader,
    device, 
    wandb_spec_table
):
    """
    Runs model on validation set and sends the first n batches of spectrogram, label, and prediction to Weights and Biases
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
        wandb_spec_data: 
            The wandb table to save the data to

    """
    with torch.no_grad():
        # only takes the first n batches
        n = 1 # maximum amount of batches to evaluate
        i = 0 # running index of example
        j = 0 # running index of val_loader
        # skip the first n images in validation loader, these have constant values due to padding

        while i < n and j < len(val_loader):
            # load file and do inference
            # iterates over val loader until file with non-constant values is found
            while True:
                x_v, y_v = next(iter(val_loader))
                j = j+1
                if torch.var(x_v) != 0:
                    break
            
            x_v, y_v = data_pipeline_val((x_v.to(device), y_v.to(device).float()))
            y_v_pred = model(x_v)

            # iterate over all slices from chunk
            for x_v_slice, y_v_slice, y_v_slice_pred in zip(x_v, y_v, y_v_pred):
                wandb_spec_table.add_data(wandb.Image(x_v_slice), print_probability_ranking(y_v_slice), print_probability_ranking(y_v_slice_pred))
            i = i+1
        wandb.log({"predictions": wandb_spec_table})

         


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
        for x_v, y_v in val_loader:
            x_v, y_v = data_pipeline_val((x_v.to(device), y_v.to(device).float()))
            y_v_pred = nn.functional.softmax(model(x_v), dim=1)
            y_val_true.append(y_v)
            y_val_pred.append(y_v_pred)
            running_val_loss += criterion(y_v_pred, y_v)
        
        y_val_true = torch.cat(y_val_true).to('cpu')
        y_val_pred = torch.cat(y_val_pred).to('cpu')
        
        val_loss = running_val_loss/len(val_loader)

        if metric != None:
            val_score = metric(y_val_pred, y_val_true)
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
    epoch :int = 0):
    print(f'epoch {epoch+1}, iteration {i}/{max_i}:\trunning loss = {train_loss:.3f}\tvalidation loss = {val_loss:.3f}\ttrain metric = {train_metric:.3f}\tvalidation metric = {val_metric:.3f}') 


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
    name: str=""
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
    
    """
    print('###########################\nStarting Training on %s \n###########################'%(device))
    log_spectrogram = True
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

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device).float()
            x, y = data_pipeline_train((x, y))
            preds = model(x)
            preds = nn.functional.softmax(preds, dim=1)
            optimizer.zero_grad()
            loss = criterion(preds, y)
            running_train_loss = running_train_loss + loss.item()
            loss.backward()
            optimizer.step()
            
            
            if i % print_every == print_every-1: 
                epoch_val_loss, epoch_val_metric = validate(model, data_pipeline_val, val_loader, device, criterion, metric)
                print_output(running_train_loss/i, epoch_train_metric, epoch_val_loss, epoch_val_metric, i,len(train_loader), epoch)


        epoch_train_loss = running_train_loss/len(train_loader)

        if print_every == -1:
            epoch_val_loss, epoch_val_metric = validate(model, data_pipeline_val, val_loader, device, criterion, metric)
            print_output(epoch_train_loss, epoch_train_metric, epoch_val_loss, epoch_val_metric,len(train_loader),len(train_loader), epoch)

        wandb_log_stats(epoch_train_loss, epoch_val_loss, epoch_val_metric)
        if log_spectrogram:
            wandb_log_spectrogram(model, data_pipeline_train, val_loader, device, wandb_spec_table)
             
        if model_saver != None:
            model_saver.save_best_model(epoch_val_loss, epoch, model, optimizer, criterion)

        train_loss.append(epoch_train_loss)
        val_loss.append(epoch_val_loss)
        val_metric.append(epoch_val_metric)

    # At the end of training: Save model and training curve
    model_saver.save_final_model(epochs, model, optimizer, criterion) 
    model_saver.save_plots(train_loss, val_loss)
                    

def collate_fn(data):
    """
    Define how the DataLoaders should batch the data
    """
    max_dim = max([d[0].shape[-1] for d in data])
    pad_x = lambda x: torch.concat([x, torch.zeros((max_dim - x.shape[-1], ))])
    return torch.stack([pad_x(d[0]) for d in data], axis=0), torch.stack([torch.tensor(d[1]) for d in data])
     

def main():

    experiment_name = "baseline_" + str(int(time.time()))
    # for pre-processing
    # splitting
    duration = 30 
    n_splits = 5
    test_split = 0.05
    num_classes = 152

    # some hyperparameters
    bs = 16 # batch size
    epochs = 300
    learning_rate = 1e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    N = -1 # number of training examples (useful for testing)

    if N != -1:
        warnings.warn(f'\n\nWarning! Using only {N} training examples!\n')

    # Load data 
    with open(f'{DATA_PATH}all_birds.json') as f:
        birds = json.load(f)

    metadata = pd.read_csv(f'{DATA_PATH}train_metadata.csv')[:N]

    # train test split
    tts = metadata.sample(frac=.05).index 
    df_val = metadata.iloc[tts]
    df_train = metadata.iloc[~tts]

    # Datasets, DataLoaders
    train_data = SimpleDataset(df_train, DATA_PATH, mode='train', labels=birds)
    val_data = SimpleDataset(df_val, DATA_PATH, mode='train', labels=birds)

    train_loader = DataLoader(train_data, batch_size=bs, num_workers=4, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=bs, num_workers=4, collate_fn=collate_fn)

    # create mode
    transforms1 = TransformApplier([nn.Identity(), SelectSplitData(duration, n_splits, offset=None)])

    wav2spec = Wav2Spec()

    transforms2 = TransformApplier([nn.Identity(), InstanceNorm()]) 

    data_pipeline_train = nn.Sequential(
        transforms1, 
        wav2spec,
        transforms2, 
    ).to(device)

    data_pipeline_val = nn.Sequential(
        transforms1, 
        wav2spec, 
        transforms2
    ).to(device) # Leaving out transforms2 since I think it will mostly be relevant for training

    # Model Architecture
    cnn = PretrainedModel(
        model_name='efficientnet_b2', 
        in_chans=1, # normally 3 for RGB-images
    )

    # Post-processing
    transforms3 = TransformApplier([SimpleAttention(cnn.get_out_dim()), RejoinSplitData(duration, n_splits)])

    output_head = OutputHead(n_in=cnn.get_out_dim() * n_splits, n_out=len(birds))

    # Model definition
    model = nn.Sequential(
        cnn,
        transforms3, 
        output_head,
        # transforms4
    ).to(device)

    optimizer = Adam(model.parameters(), lr = learning_rate)
    criterion = nn.CrossEntropyLoss(weight=None, reduction='mean')
    metric = MulticlassF1Score(
        num_classes = num_classes, # TODO check this
        topk = 1, # this means we say that we take the label with the highest probability for prediction
        average='micro' # TODO Discuss that
    ) 
    
    model_saver = ModelSaver(OUTPUT_DIR, experiment_name)

    

    config = {
    "epochs": epochs,
    "batch_size": bs,
    "learning_rate": learning_rate,
    "device": device,
    "duartion" : duration,
    "n_splits" : n_splits,
    "transforms1": transforms1,
    "transforms2": transforms2,
    "transforms3": transforms3,
    "model": model,
    "test_split" : test_split
    }

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
        print_every=10
    )

if __name__ == '__main__':
    print('Starting the training...')
    main()
