"""
Train a pipeline 
"""

from modules import * 
from modules.training.train_utils import *

from modules.training.Trainer import Trainer, Metric 
from modules import PickyScore

import torch.nn as nn
import pandas as pd 
import json
from torch.utils.data import DataLoader 
from torch.optim import Adam 
from torchmetrics.classification import MultilabelF1Score, MultilabelRecall, MultilabelPrecision
import time
import warnings

import torch_audiomentations as tam
from decouple import config


DATA_PATH = config("DATA_PATH")
SPEC_PATH = config('SPEC_PATH')
OUTPUT_DIR = config("OUTPUT_DIR")
SPLIT_PATH = config("SPLIT_PATH")


LOCAL_TEST = False
WANDB = False

def main():
    experiment_name = "baseline_" + str(int(time.time())) if not LOCAL_TEST else "local"
    # for pre-processing
    # splitting
    duration = 30 
    n_splits = 6
    test_split = 0.05 # fraction of samples for the validation dataset

    # some hyperparameters
    bs = 8 # batch size

    epochs = 300
    learning_rate = 1e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    N = 100 # number of training examples (useful for testing)

    if N != -1:
        warnings.warn(f'\n\nWarning! Using only {N} training examples!\n')

    # Load data 
    with open(f'{DATA_PATH}all_birds.json') as f:
        birds = json.load(f)

    num_classes = len(birds)

    df_train = pd.read_csv(f'{SPLIT_PATH}train_metadata.csv')[:N]
    df_val = pd.read_csv(f'{SPLIT_PATH}val_metadata.csv')[:N]

    # Datasets, DataLoaders
    #train_data = SimpleDataset(df_train, DATA_PATH, mode='train', labels=birds)
    #val_data = SimpleDataset(df_val, DATA_PATH, mode='train', labels=birds)

    train_data = SpecDataset(df_train, SPEC_PATH, mode='train', labels=birds)
    val_data = SpecDataset(df_val, SPEC_PATH, mode='train', labels=birds)    

    train_loader = DataLoader(
        train_data, 
        batch_size=bs, 
        num_workers=4, 
        collate_fn=lambda x: collate_fn(x, load_all=False, sr=100, duration=30), # defined in train_utils.py
        shuffle=True, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_data, 
        batch_size=bs, 
        num_workers=4, 
        collate_fn=lambda x: collate_fn(x, load_all=False, sr=100, duration=30), # defined in train_utils.py
        shuffle=False, 
        pin_memory=True
    )
    augment = [
            tam.Gain(
            min_gain_in_db=-15.0,
            max_gain_in_db=5.0,
            p=0.5),
            tam.PolarityInversion(p=0.5)
        ]


    # create model
    transforms1_train = TransformApplier(
        [ 
            # torch_Audiomentations(augment), # AUG
            # SelectSplitData(duration, n_splits, offset=None),
            SelectSplitData(duration, n_splits, offset=0., sr=100)
            # add more transforms here
        ]
    )

    transforms1_val = TransformApplier(
        [
            # torch_Audiomentations(augment), # AUG
            # SelectSplitData(duration, n_splits, offset=0.),
            SelectSplitData(duration, n_splits, offset=0., sr=100), 
            # add more transforms here
        ]
    )


    transforms2 = TransformApplier(
        [
            # torch_Audiomentations(augment),
            InstanceNorm()
        ]
    )

    data_pipeline_train = nn.Sequential(
        transforms1_train, 
        # wav2spec_train,
        transforms2, 
    ).to(device)

    data_pipeline_val = nn.Sequential(
        transforms1_val, 
        # wav2spec_val,
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

    optimizer = Adam(model.parameters(), lr=learning_rate)

    criterion = nn.BCELoss() 

    metric_f1micro = MultilabelF1Score(
        num_labels = num_classes, 
        topk = 1, # this means we say that we take the label with the highest probability for prediction
        average='micro', 
    ).to(device) 

    metric_f1_ours = PickyScore(MultilabelF1Score)
    metric_recall_ours = PickyScore(MultilabelRecall)
    metric_prec_ours = PickyScore(MultilabelPrecision)

    metrics = {
                'F1Micro': metric_f1micro,
                'F1Ours': metric_f1_ours,
                'RecallOurs': metric_recall_ours,
                'PrecisionOurs': metric_prec_ours, 
            }
    model_saver = ModelSaver(OUTPUT_DIR, experiment_name)

    overlap = .3
    validator = Validator(data_pipeline_val, model, overlap=overlap, device=device)

    metrics = [
        Metric(name, method) for name, method in metrics.items()
    ]

    

    config = {
        "epochs": epochs,
        "batch_size": bs,
        "learning_rate": learning_rate,
        "device": device,
        "duration" : duration,
        "n_splits" : n_splits,
        "overlap": overlap, 
        "transforms1_train": transforms1_train,
        "transforms1_val": transforms1_val,
        "transforms2": transforms2,
        "transforms3": transforms3,
        "model": model,
        "test_split" : test_split
    }

    trainer = Trainer(
        model=model, 
        data_pipeline_train=data_pipeline_train, 
        data_pipeline_val=data_pipeline_val, 
        model_saver=model_saver,
        validator=validator,
        criterion=criterion, 
        optimizer=optimizer, 
        device=device, 
        metrics=metrics, 
        validate_every=10, 
        use_wandb=WANDB, 
        wandb_args={
            'columns': ['Predicted', 'Expected'], 
            'project_name': 'Baseline', 
            'experiment_name': experiment_name, 
            'config': config, 
        }
    )
    
    trainer.train(
        train_loader, 
        val_loader, 
        epochs=epochs
    )


if __name__ == '__main__':
    main()
