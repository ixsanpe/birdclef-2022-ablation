

"""
Train a pipeline 
"""

from modules import * 
from modules.training.train_utils import *

from modules.training.Trainer import Trainer, Metric
from modules.training.FocalLoss import FocalLoss 
from modules.training.WeightedBCELoss import WeightedBCELoss
from modules.training.WeightedFocalLoss import WeightedFocalLoss

from modules import PickyScore
from modules import RecallMacro, PrecisionMacro, F1Macro


import argparse
from ablation import s2b, check_args

import torch.nn as nn
import pandas as pd 
import json
from torch.utils.data import DataLoader 
from torch.optim import Adam 
from torchmetrics.classification import MultilabelF1Score, MultilabelRecall, MultilabelPrecision
import warnings

import torch_audiomentations as tam
from decouple import config as cfg
from datetime import datetime

DATA_PATH = cfg("DATA_PATH")
SPEC_PATH = cfg('SPEC_PATH')
OUTPUT_DIR = cfg("OUTPUT_DIR")

def parse_args():
    """
    This script runs with args. Below is a function to define the arguments
    and to parse them as they come in. 
    """
    parser = argparse.ArgumentParser(description='Start a training run')
    # Initialize the default boolean parameter
    # parser.add_argument('--default_bool', type=s2b)
    # args = parser.parse_known_args() 
    # default_bool = args[0].default_bool

    # Training hyperparameters
    parser.add_argument('--test_split', type=float, default=.05, help='fraction of samples for the validation dataset')
    parser.add_argument('--batch_size_train', type=int, default=16)
    parser.add_argument('--batch_size_val', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--N', type=int, default=-1, help='number of samples used for training') #-1
    parser.add_argument('--loss', type=str, default='BCELoss')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--overlap', type=float, default=.3)
    parser.add_argument('--validate_every', type=int, default=-1)
    parser.add_argument('--precompute', type=s2b, default='True')
    parser.add_argument('--augs', type=str, default='', help='Name of Augmentation; Possible choices: gain, gaussiannoise, timestrecht, pitchshift, shift, backgroundnoise. Only available for precompute=True')
    parser.add_argument('--aug_prob', type=float, default=1.0)
    parser.add_argument('--policy', type=str, default='max_all', help='strategy to aggregate preds for validation')
    parser.add_argument('--scheme', type=str, default='old', help='new scheme attempted to speed up   validator but seems buggy')
    
    # Pipeline configuration
    parser.add_argument('--duration', type=int, default=1000, help='duration to train on')
    parser.add_argument('--max_duration', type=int, default=1000, help='how much of the data to load')
    parser.add_argument('--sr', type=float, default=1, help='(effective) sample rate')
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--offset_val', type=float, default=0.)
    parser.add_argument('--offset_train', type=int, default=None)
    parser.add_argument('--model_name', type=str, default='efficientnet_b2')
    parser.add_argument('--InstanceNorm', type=s2b)
    parser.add_argument('--SimpleAttention', type=s2b, default='True')
    parser.add_argument('--k_runs', type=int, choices=[1, 3], default=1, help='Number of splits for cross validation. Default to 1. Current implementation only support 1 and 3')

    # wandb stuff
    parser.add_argument('--wandb', type=s2b, default='True')
    parser.add_argument('--project_name', type=str, default='AblationTest')
    parser.add_argument('--experiment_name', type=str, default='baseline_'+ datetime.now().strftime("%Y-%m-%d-%H-%M"))

    return parser.parse_args()

def train(args, k=1):
    experiment_name = args.experiment_name
    # Take args defined in parse_args(): 
    duration = args.duration 
    max_duration = args.max_duration
    n_splits = args.n_splits
    test_split = args.test_split 
    sr = args.sr 
    offset_train = args.offset_train 
    offset_val = args.offset_val

    bs_train = args.batch_size_train
    bs_val = args.batch_size_val
    precompute = args.precompute
    augs = args.augs
    aug_prob = args.aug_prob

    epochs = args.epochs 
    learning_rate = args.learning_rate
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = args.model_name

    # Here, to use the args, define a dict
    # containing losses available, and access them with the
    # key args.loss

    losses = {'BCELoss':nn.BCELoss(),'FocalLoss':FocalLoss(), 'WeightedBCELoss':WeightedBCELoss(),'WeightedFocalLoss':WeightedFocalLoss()}


    criterion = losses[args.loss]

    # Similarly define optimizers as losses
    optimizers = {'Adam': Adam}
    optimizer = optimizers[args.optimizer]

    overlap = args.overlap
    validate_every = args.validate_every
    use_wandb = args.wandb
    project_name = args.project_name 

    N = args.N 

    # From this point on, it is a regular train function
    if N != -1:
        warnings.warn(f'\n\nWarning! Using only {N} training examples!\n')

    # Load data 
    with open(f'{DATA_PATH}all_birds.json') as f:
        birds = json.load(f)

    num_classes = len(birds)

    if args.k_runs == 1:
        SPLIT_PATH = cfg("SPLIT_PATH")
        df_train = pd.read_csv(f'{SPLIT_PATH}train_metadata.csv')[:N]
        df_val = pd.read_csv(f'{SPLIT_PATH}val_metadata.csv')[:N]
    elif args.k_runs == 3:
        SPLIT_PATH_KFOLD = cfg("SPLIT_PATH_KFOLD")
        df_train = pd.read_csv(f'{SPLIT_PATH_KFOLD}{k}/train_metadata.csv')[:N]
        df_val = pd.read_csv(f'{SPLIT_PATH_KFOLD}{k}/val_metadata.csv')[:N]
    else:
        raise NotImplementedError(f'k-fold cross validation only implemented for k =1, 3 but got {k=}')

    # Datasets, DataLoaders
    if precompute:
        if augs != '':
            AUGMENT_PATH  = cfg("AUGMENT_PATH")
            train_data = AugmentDataset(df_train, SPEC_PATH, AUGMENT_PATH, augmentations = [augs], mode='train', labels=birds, augment_prob=aug_prob)
            val_data = SpecDataset(df_val, SPEC_PATH, mode='train', labels=birds) 
        else:    
            train_data = SpecDataset(df_train, SPEC_PATH, mode='train', labels=birds)
            val_data = SpecDataset(df_val, SPEC_PATH, mode='train', labels=birds)  

    else:
        if augs != '':
            print('\n Warning: Augmentations cannot be applied if precomputed is not activated \n')

        train_data = SimpleDataset(df_train, DATA_PATH, mode='train', labels=birds)
        val_data = SimpleDataset(df_val, DATA_PATH, mode='train', labels=birds)


    train_selector = Selector(duration=max_duration, offset=offset_train, device='cpu') 

    train_loader = DataLoader(
        train_data, 
        batch_size=bs_train, 
        num_workers=8, #8
        collate_fn=lambda x: collate_fn(x, load_all=False, sr=sr, duration=max_duration, selector=train_selector), # defined in train_utils.py
        shuffle=True, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_data, 
        batch_size=bs_val, 
        num_workers=8, #8
        collate_fn=lambda x: collate_fn(x, load_all=True), # defined in train_utils.py
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
                p=0.5
            ),
            tam.PolarityInversion(p=0.5)
        ]


    transforms2 = [ 
        InstanceNorm()
    ]

    transforms2 = TransformApplier(
        check_args(transforms2, args)
    )


    wav2spec = nn.Identity() if precompute else Wav2Spec()

    data_pipeline_train = nn.Sequential(
        transforms1_train, 
        wav2spec,
        transforms2, 
    ).to(device)

    data_pipeline_val = nn.Sequential(
        transforms1_val, 
        wav2spec,
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

    metric_f1_ours = F1Macro() # PickyScore(MultilabelF1Score)
    metric_recall_ours = RecallMacro() # PickyScore(MultilabelRecall)
    metric_prec_ours = PrecisionMacro() # PickyScore(MultilabelPrecision)
    #metric_f1_old = PickyScore(MultilabelF1Score)
    #metric_recall_old = PickyScore(MultilabelRecall)
    #metric_prec_old = PickyScore(MultilabelPrecision).to(device)

    metrics = {
                'F1Micro': metric_f1micro,
                'F1Ours': metric_f1_ours,
                'RecallOurs': metric_recall_ours,
                'PrecisionOurs': metric_prec_ours,
                #'F1_old': metric_f1_old,
                #'Recall_old': metric_recall_old,
                #'Precision_old': metric_prec_old,
            }

    model_saver = ModelSaver(OUTPUT_DIR, experiment_name)

    policies = {
        'first_and_final': first_and_final,
        'max_thresh': max_thresh, 
        'max_all': max_all, 
        'rolling_avg': rolling_avg,     
    }
    policy = policies[args.policy]

    validator = Validator(
        data_pipeline_val, 
        model, 
        overlap=overlap, 
        device=device, 
        policy=policy,
        scheme=args.scheme
    )

    metrics = [
        Metric(name, method) for name, method in metrics.items()
    ]

    # config = {
    #     "epochs": epochs,
    #     "batch_size_train": bs_train,
    #     "batch_size_val": bs_val,
    #     "learning_rate": learning_rate,
    #     "device": device,
    #     "duration" : duration,
    #     "n_splits" : n_splits,
    #     "overlap": overlap, 
    #     "transforms1_train": transforms1_train,
    #     "transforms1_val": transforms1_val,
    #     "transforms2": transforms2,
    #     "transforms3": transforms3,
    #     "model": model,
    #     "test_split" : test_split, 
    #     "args": args
    # }
    config = vars(args)

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
            'group': None
        }
    )

    trainer.train(
        train_loader, 
        val_loader, 
        epochs=epochs
    )


def main():
    args = parse_args()
    print(args)

    for k in range(args.k_runs):
        if args.k_runs > 1:
            print('#' * 20 + f'\nRunning {k} split\n' + '#' * 20)
        train(args, k)


if __name__ == '__main__':
    main()