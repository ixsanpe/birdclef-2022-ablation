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

import torch.nn as nn
import pandas as pd 
import json
from torch.utils.data import DataLoader 
from torch.optim import Adam 
from typing import Callable
import warnings

DATA_PATH = 'birdclef-2022/'

def show_progress(
    model, 
    data_pipeline_val, 
    val_loader, 
    device, 
    criterion, 
    running_loss, 
    epoch, 
    i, 
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
        i:
            current iteration in the epoch for which to report progress
    """
    with torch.no_grad():
        val_loss = 0.
        for j, (x_v, y_v) in enumerate(val_loader):
            x_v, y_v = data_pipeline_val((x_v.to(device), y_v.to(device).float()))
            val_loss += criterion(model(x_v), y_v)
        
    print(f'epoch {epoch+1}, iteration {i}:\trunning loss = {running_loss/(i+1):.3f}\tvalidation loss = {val_loss/(j+1):.3f}') 

def train(
    model: nn.Module, 
    data_pipeline_train: nn.Module, 
    data_pipeline_val: nn.Module, 
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    optimizer: torch.optim.Optimizer, 
    criterion: Callable, 
    epochs: int=1,
    print_every: int=-1, 
    device: str='cpu'
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
        epochs:
            number of epochs for which to train
        print_every:
            how often to report progress. If print_every==-1, we print at the end of every epoch.
        device:
            device on which to train
    
    """
    
    for epoch in range(epochs):
        print(f'starting epoch {epoch+1} / {epochs}')
        running_loss = 0.
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device).float()
            x, y = data_pipeline_train((x, y))
            preds = model(x)
            optimizer.zero_grad()
            loss = criterion(preds, y)
            running_loss = running_loss + loss.item()
            loss.backward()
            optimizer.step()
            if i % print_every == print_every-1:
                show_progress(model, data_pipeline_val, val_loader, device, criterion, running_loss, epoch, i)
        if print_every == -1:
            show_progress(model, data_pipeline_val, val_loader, device, criterion, running_loss, epoch, i)    
            
                    

def collate_fn(data):
    """
    Define how the DataLoaders should batch the data
    """
    max_dim = max([d[0].shape[-1] for d in data])
    pad_x = lambda x: torch.concat([x, torch.zeros((max_dim - x.shape[-1], ))])
    return torch.stack([pad_x(d[0]) for d in data], axis=0), torch.stack([torch.tensor(d[1]) for d in data])
     

def main():
    # for pre-processing
    # splitting
    duration = 30 
    n_splits = 5

    # some hyperparameters
    bs = 2 # batch size
    epochs = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    N = -1 # number of training examples (useful for testing)

    if N != -1:
        warnings.warn(f'\n\nWarning! Using only {N} training examples!\n')

    # Load data 
    with open(f'{DATA_PATH}scored_birds.json') as f:
        birds = json.load(f)

    metadata = pd.read_csv(f'{DATA_PATH}train_metadata.csv')[:N]

    # train test split
    tts = metadata.sample(frac=.05).index 
    df_val = metadata.iloc[tts]
    df_train = metadata.iloc[~tts]

    # Datasets, DataLoaders
    train_data = SimpleDataset(df_train, DATA_PATH, mode='train', labels=birds)
    val_data = SimpleDataset(df_val, DATA_PATH, mode='train', labels=birds)

    train_loader = DataLoader(train_data, batch_size=bs, num_workers=8, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=bs, num_workers=8, collate_fn=collate_fn)

    # create mode
    transforms1 = TransformApplier([SelectSplitData(duration, n_splits)])

    wav2spec = Wav2Spec()

    transforms2 = TransformApplier([nn.Identity()])

    data_pipeline_train = nn.Sequential(
        transforms1, 
        wav2spec,
        transforms2, 
    ).to(device)

    data_pipeline_val = nn.Sequential(
        transforms1, 
        wav2spec, 
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

    optimizer = Adam(model.parameters(), )
    criterion = nn.CrossEntropyLoss()

    train(
        model, 
        data_pipeline_train,
        data_pipeline_val,  
        train_loader, 
        val_loader, 
        optimizer, 
        criterion, 
        epochs=epochs, 
        device=device, 
        print_every=10, 
    )

if __name__ == '__main__':
    main()
