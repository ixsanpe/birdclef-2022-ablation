"""
Train a pipeline 
"""

from modules import * 
from modules.training.train_utils import *

from modules.training.Trainer import Trainer, Metric 

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
OUTPUT_DIR = config("OUTPUT_DIR")


LOCAL_TEST = False
WANDB = True


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
    learning_rate = 1e-4

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    N = 200 # number of training examples (useful for testing)

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
        collate_fn=collate_fn, # defined in train_utils.py
        shuffle=True, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_data, 
        batch_size=bs, 
        num_workers=4, 
        collate_fn=collate_fn, # defined in train_utils.py
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

    augment = [
            tam.Gain(
            min_gain_in_db=-15.0,
            max_gain_in_db=5.0,
            p=0.5),
            tam.PolarityInversion(p=0.5)
        ]

    transforms2 = TransformApplier(
        [
            # torch_Audiomentations(augment), 
            InstanceNorm()
        ]
    )
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

    criterion = nn.BCELoss() 

    metric_f1micro = MultilabelF1Score(
        num_labels = num_classes, # TODO check this
        topk = 1, # this means we say that we take the label with the highest probability for prediction
        average='micro' # TODO Discuss that
    ).to(device) 

    metric_f1macro = MultilabelF1Score(
        num_labels = num_classes, # TODO check this
        topk = 1, # this means we say that we take the label with the highest probability for prediction
        average='macro' # TODO Discuss that
    ).to(device) 
    metric_recall = MultilabelRecall( 
        num_labels=num_classes,
        average='macro'
    ).to(device)  # Gives a better idea since most predictions are 0 anyways?
    

    metric_precision = MultilabelPrecision( 
        num_labels=num_classes,
        average='macro'
    ).to(device)  

    metrics = {'F1Micro': metric_f1micro,
                'F1Macro': metric_f1macro,
                'Recall': metric_recall,
                'Precision': metric_precision}
    model_saver = ModelSaver(OUTPUT_DIR, experiment_name)

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
        "transforms1": transforms1,
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
