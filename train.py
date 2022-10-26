"""
Train a pipeline 
"""
from torch import Tensor
from modules.PretrainedModel import * 
from modules.OnlyXTransform import *
from modules.Wav2Spec import * 
from modules.TransformApplier import * 
from modules.SimpleDataset import * 
from modules.SimpleAttention import * 
from modules.SelectSplitData import *
from modules.Normalization import *
from modules.model_utils import *
from modules.Audiomentations import *
from modules.torch_Audiomentations import *
from modules.Postprocessing import *

import torch.nn as nn
import pandas as pd 
import json
from torch.utils.data import DataLoader 
from torch.optim import Adam 
from torchvision.utils import make_grid
from typing import Callable
from torchmetrics.classification import MultilabelF1Score, MultilabelRecall, MultilabelPrecision
import wandb
import time
import warnings
import os 
import torch_audiomentations  as tam
import audiomentations as am

DATA_PATH = os.getcwd() + '/birdclef-2022/'
OUTPUT_DIR = 'output/'


def validate(
    model: nn.Module, 
    data_pipeline_val : nn.Module, 
    val_loader : DataLoader, 
    device: str, 
    criterion: Callable, 
    metrics = None,
    pred_threshold :Callable= PredictionThreshold(0.5)
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
        metrics:
            dict of metrics. Form is {'name_of_metric': metric:Callable,...}
        pred_threshold:
            a callable. takes a input a float prediction tensor and outputs a thresholded tensor of the same shape
        
    """
    with torch.no_grad():
        y_val_true = []
        y_val_pred = []
        running_val_loss = 0.
        for x_v, y_v in val_loader:
            x_v, y_v = data_pipeline_val((x_v.to(device), y_v.to(device).float()))
            y_v_logits = model(x_v)
            y_v_pred = torch.sigmoid(y_v_logits)
            y_val_true.append(y_v)
            y_val_pred.append(y_v_pred)
            # running_val_loss += criterion(y_v_logits, y_v)
            running_val_loss += criterion(y_v_pred, y_v)
        
        y_val_true = torch.cat(y_val_true).to('cpu')
        y_val_pred = torch.cat(y_val_pred).to('cpu')
        
        val_loss = running_val_loss/len(val_loader)
        
        val_scores = dict()
        if metrics:
            for me_name, me_func in metrics.items():
                try:
                    score = me_func(pred_threshold(y_val_pred), y_val_true.int())
                except  Exception as ex:
                    print(f'Exception {ex} in metric {me_name}')
                    score = 0.
                val_scores['val_' + me_name] = score

    return val_loss, val_scores



def train(
    model: nn.Module, 
    data_pipeline_train: nn.Module, 
    data_pipeline_val: nn.Module, 
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    optimizer: torch.optim.Optimizer, 
    criterion: Callable, 
    metrics: dict, 
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
        metrics:
            a dict of metrics for the prediction
        epochs:
            number of epochs for which to train
        print_every:
            how often to report progress. If print_every==-1, we print at the end of every epoch.
        device:
            device on which to train
        n_splits:
            the number of splits of the 30s recording. Essential becasue y_true has shape (B,C), whereas y_pred and x have shape (B*n_splits, C) and (B*n_splits, X, Y)
    
    """
    print('###########################\nStarting Training on %s \n###########################'%(device))
    if log_spectrogram:
        wandb_spec_table = wandb.Table(columns=['Sprectrogram', 'Predicted', 'Expected'])

    train_loss, val_loss, val_metric = [], [], []
    for epoch in range(epochs):
        print(f'starting epoch {epoch+1} / {epochs}')
        running_train_loss = 0.
        epoch_train_loss = 0.
        epoch_train_metrics = None # not used at the moment
        epoch_val_loss = 0.
        epoch_val_metrics = None

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device).float()
            x, y = data_pipeline_train((x, y))
            logits = model(x)
            y_pred = torch.sigmoid(logits)
            optimizer.zero_grad()
            loss = criterion(y_pred, y)
            running_train_loss = running_train_loss + loss.item()
            loss.backward()
            optimizer.step()
            
            
            if i % print_every == print_every-1: 
                epoch_val_loss, epoch_val_metrics = validate(model, data_pipeline_val, val_loader, device, criterion, metrics)
                print_output(running_train_loss/i, loss.item() ,  epoch_train_metrics, epoch_val_loss, epoch_val_metrics, i,len(train_loader), epoch)


        epoch_train_loss = running_train_loss/len(train_loader)

        if print_every == -1:
            epoch_val_loss, epoch_val_metrics = validate(model, data_pipeline_val, val_loader, device, criterion, metrics)
            print_output(epoch_train_loss, epoch_train_metrics, epoch_val_loss, epoch_val_metrics, len(train_loader),len(train_loader), epoch)

        wandb_log_stats(epoch_train_loss, epoch_val_loss, epoch_val_metrics)
        if log_spectrogram:
            wandb_log_spectrogram(model, data_pipeline_train, val_loader, device, wandb_spec_table, n_splits)
             
        if model_saver != None:
            model_saver.save_best_model(epoch_val_loss, epoch, model, optimizer, criterion)

        train_loss.append(epoch_train_loss)
        val_loss.append(epoch_val_loss)
        val_metric.append(epoch_val_metrics)

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

    # some hyperparameters
    bs = 8 # batch size
    epochs = 300
    learning_rate = 1e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    N = -1 # number of training examples (useful for testing)

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

    train_loader = DataLoader(train_data, batch_size=bs, num_workers=4, collate_fn=collate_fn, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=bs, num_workers=4, collate_fn=collate_fn, shuffle=False, pin_memory=True)


    # create model
    transforms1 = TransformApplier([nn.Identity(), SelectSplitData(duration, n_splits, offset=0)])

    wav2spec = Wav2Spec()

    transforms2 = TransformApplier([nn.Identity(), InstanceNorm()]) 
    """
    augment=[ # TODO find the right augmentations from torch_audiomentations 
    am.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    am.TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    am.PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    am.Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
    ]
    transforms2 = TransformApplier([Audiomentations(augment), InstanceNorm()]) 
    """
    augment = [tam.Gain(
            min_gain_in_db=-15.0,
            max_gain_in_db=5.0,
            p=0.5),
            tam.PolarityInversion(p=0.5)]
    transforms2 = TransformApplier([torch_Audiomentations(augment), InstanceNorm()])
    #TODO: audiomentations has better transformations than torch.audiomentations, do we find a way to use it on gpu?
    
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
    transforms3 = TransformApplier([SimpleAttention(cnn.get_out_dim()), RejoinSplitData(duration, n_splits)])

    output_head = OutputHead(n_in=cnn.get_out_dim() * n_splits, n_out=num_classes)

    # Model definition
    model = nn.Sequential(
        cnn,
        transforms3, 
        output_head,
        # transforms4
    ).to(device)

    optimizer = Adam(model.parameters(), lr = learning_rate)
    criterion = nn.BCELoss() # nn.CrossEntropyLoss(weight=None, reduction='mean')
    metric_f1micro = MultilabelF1Score(
        num_labels = num_classes, # TODO check this
        topk = 1, # this means we say that we take the label with the highest probability for prediction
        average='micro' # TODO Discuss that
    ) 

    metric_f1macro = MultilabelF1Score(
        num_labels = num_classes, # TODO check this
        topk = 1, # this means we say that we take the label with the highest probability for prediction
        average='macro' # TODO Discuss that
    ) 
    metric_recall = MultilabelRecall( 
        num_labels=num_classes,
        average='macro'
    ) # Gives a better idea since most predictions are 0 anyways?
    

    metric_precision = MultilabelPrecision( 
        num_labels=num_classes,
        average='macro'
    ) 
    metrics = {'F1Micro': metric_f1micro,
                'F1Macro': metric_f1macro,
                'Recall': metric_recall,
                'Precision': metric_precision}
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
        metrics,
        model_saver=model_saver,
        epochs=epochs, 
        device=device, 
        print_every=10,
        n_splits=n_splits
    )

if __name__ == '__main__':
    print('Starting the training...')
    main()
