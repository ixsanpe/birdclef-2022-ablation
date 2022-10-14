"""
Train a pipeline 
"""

from modules.PretrainedModel import * 
from modules.OnlyXTransform import *
from modules.Wav2Spec import * 
from modules.TransformApplier import * 
from modules.SimpleDataset import * 
from modules.SimpleAttention import * 

import torch.nn as nn
import numpy as np
import pandas as pd 
import json
from torch.utils.data import DataLoader 
from torch.optim import Adam 


DATA_PATH = 'birdclef-2022/'

def train(model, train_loader, val_loader, optimizer, criterion, epochs=1, print_every=10, device='cpu'):
    for epoch in range(epochs):
        print(f'starting epoch {epoch+1} / {epochs}')
        running_loss = 0.
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device).float()
            preds = model(x)
            optimizer.zero_grad()
            loss = criterion(preds, y)
            running_loss = running_loss + loss.item()
            loss.backward()
            optimizer.step()
            if i % print_every == print_every-1:
                with torch.no_grad():
                    val_loss = 0.
                    for j, (x_v, y_v) in enumerate(val_loader):
                        val_loss += criterion(model(x_v), y_v.float())
                    
                print(f'epoch {epoch+1}, iteration {i}:\trunning loss = {running_loss/(i+1):.3f}\tvalidation loss = {val_loss/(j+1):.3f}')
                    

def collate_fn(data):
    max_dim = 1600000
    pad_x = lambda x: torch.concat([x, torch.zeros((max_dim - x.shape[-1], ))])
    return torch.stack([pad_x(d[0]) for d in data], axis=0), torch.stack([torch.tensor(d[1]) for d in data])
     

def main():
    # some hyperparameters
    bs = 2 # batch size
    epochs = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load data 
    with open(f'{DATA_PATH}scored_birds.json') as f:
        birds = json.load(f)

    metadata = pd.read_csv(f'{DATA_PATH}train_metadata.csv')[:200]
    tts = metadata.sample(frac=.2).index # train test split
    df_val = metadata.iloc[tts]
    df_train = metadata.iloc[~tts]

    train_data = SimpleDataset(df_train, DATA_PATH, mode='train', labels=birds)
    val_data = SimpleDataset(df_val, DATA_PATH, mode='train', labels=birds)

    train_loader = DataLoader(train_data, batch_size=bs, num_workers=8, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=bs, num_workers=8, collate_fn=collate_fn)

    # create mode
    transforms1 = TransformApplier([nn.Identity()])

    wav2spec = Wav2Spec()

    transforms2 = TransformApplier([OnlyXTransform()])

    cnn = PretrainedModel(
        model_name='efficientnet_b2', 
        in_chans=1, # normally 3 for RGB-images
    )

    transforms3 = TransformApplier([SimpleAttention(cnn.get_out_dim())])

    output_head = OutputHead(n_in=cnn.get_out_dim(), n_out=21)

    model = nn.Sequential(
        transforms1, 
        wav2spec,
        transforms2, 
        cnn,
        transforms3, 
        output_head,
    ).to(device)

    optimizer = Adam(model.parameters(), )
    criterion = nn.CrossEntropyLoss()

    train(
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        criterion, 
        epochs=epochs, 
        device=device, 
    )

if __name__ == '__main__':
    main()
