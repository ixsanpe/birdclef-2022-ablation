"""
Train a pipeline 
"""

from modules import * 
from modules.training.train_utils import *

from modules.training.Trainer import Trainer, Metric 
from modules import PickyScore

import argparse
from ablation import s2b, check_args

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

def parse_args():
    parser = argparse.ArgumentParser(description='Start a training run')
    # Initialize the default boolean parameter
    parser.add_argument('--default_bool', type=s2b)
    args = parser.parse_known_args() 
    default_bool = args[0].default_bool

    # Training hyperparameters
    parser.add_argument('--test_split', type=float, default=.05, help='fraction of samples for the validation dataset')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--N', type=int, default=-1, help='number of samples used for training')
    parser.add_argument('--loss', type=str, default='BCELoss')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--overlap', type=float, default=.3)
    parser.add_argument('--validate_every', type=int, default=150)

    # Pipeline configuration
    parser.add_argument('--duration', type=int, default=30, help='duration to train on')
    parser.add_argument('--max_duration', type=int, default=30, help='how much of the data to load')
    parser.add_argument('--sr', type=float, default=10)
    parser.add_argument('--n_splits', type=int, default=6)
    parser.add_argument('--offset_val', type=float, default=0.)
    parser.add_argument('--offset_train', type=int, default=None)
    parser.add_argument('--model_name', type=str, default='efficientnet_b2')
    parser.add_argument('--InstanceNorm', type=s2b, default=default_bool)
    parser.add_argument('--SimpleAttention', type=s2b, default=default_bool)

    # wandb stuff
    parser.add_argument('--wandb', type=s2b, default='True')
    parser.add_argument('--project_name', type=str, default='Baseline')
    parser.add_argument('--experiment_name', type=str, default='baseline_'+str(int(time.time())))

    return parser.parse_args()



def main():
    args = parse_args()
    print(args)
    experiment_name = args.experiment_name
    # for pre-processing
    # splitting
    duration = args.duration 
    max_duration = args.max_duration
    n_splits = args.n_splits
    test_split = args.test_split 
    sr = args.sr 
    offset_train = args.offset_train 
    offset_val = args.offset_val

    # some hyperparameters
    bs = args.batch_size

    epochs = args.epochs 
    learning_rate = args.learning_rate
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = args.model_name

    losses = {'BCELoss': nn.BCELoss()}
    criterion = losses[args.loss]

    optimizers = {'Adam': Adam}
    optimizer = optimizers[args.optimizer]

    overlap = args.overlap
    validate_every = args.validate_every
    use_wandb = args.wandb
    project_name = args.project_name 

    N = args.N 

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
    # train_data = SimpleDataset(df_train, DATA_PATH, mode='train', labels=birds)
    # val_data = SimpleDataset(df_val, DATA_PATH, mode='train', labels=birds)

    train_data = SpecDataset(df_train, SPEC_PATH, mode='train', labels=birds)
    val_data = SpecDataset(df_val, SPEC_PATH, mode='train', labels=birds)    

    train_loader = DataLoader(
        train_data, 
        batch_size=bs, 
        num_workers=8, 
        collate_fn=lambda x: collate_fn(x, load_all=False, sr=sr, duration=max_duration), # defined in train_utils.py
        shuffle=True, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_data, 
        batch_size=bs, 
        num_workers=8, 
        collate_fn=lambda x: collate_fn(x, load_all=False, sr=sr, duration=max_duration), # defined in train_utils.py
        shuffle=False, 
        pin_memory=True
    )


    # create model
    transforms1_train = [ 
            SelectSplitData(duration, n_splits, offset=offset_train, sr=sr)
            # add more transforms here
        ]
    transforms1_train = TransformApplier(
        check_args(transforms1_train, args, keep='SelectSplitData')
    )

    transforms1_val = [ 
            SelectSplitData(duration, n_splits, offset=offset_val, sr=sr), 
            # add more transforms here
        ]
    transforms1_val = TransformApplier(
        check_args(transforms1_val, args, keep='SelectSplitData')
    )

    augment = [
            tam.Gain(
            min_gain_in_db=-15.0,
            max_gain_in_db=5.0,
            p=0.5),
            tam.PolarityInversion(p=0.5)
        ]

    transforms2 = [ 
        InstanceNorm()
    ]
    
    transforms2 = TransformApplier(
        check_args(transforms2, args)
    )
    #TODO: audiomentations has better transformations than torch.audiomentations, do we find a way to use it on gpu?
    
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
        model_name=model_name, 
        in_chans=1, # normally 3 for RGB-images
    )

    # Post-processing
    transforms3 = [
            SimpleAttention(cnn.get_out_dim()), 
            RejoinSplitData(duration, n_splits, sr=sr)
        ]
    transforms3 = TransformApplier(
        check_args(transforms3, args, keep=['SimpleAttention', 'RejoinSplitData'])
    )

    output_head = OutputHead(n_in=cnn.get_out_dim() * n_splits, n_out=num_classes)

    # Model definition
    model = nn.Sequential(
        cnn,
        transforms3, 
        output_head,
    ).to(device)

    optimizer = optimizer(model.parameters(), lr=learning_rate)

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
        validate_every=validate_every, 
        use_wandb=use_wandb, 
        wandb_args={
            'columns': ['Predicted', 'Expected'], 
            'project_name': project_name, 
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
