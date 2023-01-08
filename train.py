

"""
Train a pipeline 
"""

from modules import * 
from modules.training.train_utils import *

from modules.training.Trainer import Trainer, Metric
from modules.training.FocalLoss import FocalLoss 
from modules.training.WeightedBCELoss import WeightedBCELoss
from modules.training.WeightedFocalLoss import WeightedFocalLoss

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

    # Training hyperparameters and configuration
    parser.add_argument('--batch_size_train', type=int, default=16, help='batch size for train dataloader')
    parser.add_argument('--batch_size_val', type=int, default=1, help='batch size for validation data loader')
    parser.add_argument('--epochs', type=int, default=10, help='the number of epochs for which to train')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='the learning rate for the optimizer')
    parser.add_argument('--N', type=int, default=-1, help='number of samples used for training') 
    parser.add_argument('--loss', type=str, default='BCELoss', help='the criterion to optimize')
    parser.add_argument('--optimizer', type=str, default='Adam', help='the optimizer with which to train the model')
    parser.add_argument('--overlap', type=float, default=.3, help='the amount by which segments overlap when splitting a long recording into segments')
    parser.add_argument('--validate_every', type=int, default=-1, help='how often to validate (in training iterations). \
                                                                            If -1, we validate at the end of every epoch')
    parser.add_argument('--precompute', type=s2b, default='True', help='whether to precompute the spectrograms, which offers a significant speedup')
    parser.add_argument('--augs', type=str, default='', help='Name of Augmentation; Possible choices: gain, gaussiannoise, timestretch, pitchshift, shift, backgroundnoise. Only available for precompute=True')
    parser.add_argument('--aug_prob', type=float, default=1.0, help='The amount of augmented data in relation to the size of the normal dataset')
    parser.add_argument('--policy', type=str, default='max_all', help='strategy to aggregate preds for validation')
    parser.add_argument('--scheme', type=str, default='new', help='if "new", it batches the segments during validation, if "old", it does not')
    
    # Pipeline configuration
    parser.add_argument('--duration', type=int, default=1500, help='duration to train on (500 corresponds to 10 seconds if we precompute)')
    parser.add_argument('--max_duration', type=int, default=1500, help='how much of the data to load before selecting a duration')
    parser.add_argument('--sr', type=float, default=1, help='(effective) sample rate; 1 if we use a spectrogram, and otherwise the actual sample rate')
    parser.add_argument('--n_splits', type=int, default=5, help='when splitting the data so that we could apply mixup, this is the number of splits \
                                                                    for each loaded recording')
    parser.add_argument('--offset_val', type=float, default=0., help='where to start validating in the recording. if None, we choose a random\
                                                                        starting point')
    parser.add_argument('--offset_train', type=int, default=None, help='where to start the training sample in the recording. if None, we choose a random\
                                                                        starting point')
    parser.add_argument('--model_name', type=str, default='efficientnet_b2', help='the name of the model architecture provided by timm')
    parser.add_argument('--InstanceNorm', type=s2b, default='False', help='whether to use Instance Normalization or not')
    parser.add_argument('--SimpleAttention', type=s2b, default='True', help='whether to use SimpleAttention or not. Currently, if False, this is overridden')
    parser.add_argument('--k_runs', type=int, choices=[1, 3], default=1, help='Number of splits for cross validation. Default to 1. Current implementation only supports 1 and 3')

    # Weights and Biases related parameters
    parser.add_argument('--wandb', type=s2b, default='True', help='whether to log the run on weights and biases')
    parser.add_argument('--project_name', type=str, default='AblationTest', help='the project name on weights and biases')
    parser.add_argument('--experiment_name', type=str, default='baseline_'+ datetime.now().strftime("%Y-%m-%d-%H-%M"), help='the \
                                                                                experiment name on weights and biases')

    return parser.parse_args()

def train(args, k=1):
    experiment_name = args.experiment_name + f'_{k=}'
    # Take args defined in parse_args(): 
    duration = args.duration 
    max_duration = args.max_duration
    n_splits = args.n_splits
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
    losses = {
        'BCELoss':nn.BCELoss(),
        'FocalLoss':FocalLoss(),
        'WeightedBCELoss_beta_0.85':WeightedBCELoss(beta=0.85),
        'WeightedBCELoss_beta_0.875':WeightedBCELoss(beta=0.875),
        'WeightedBCELoss_beta_0.9':WeightedBCELoss(beta=0.9), 
        'WeightedBCELoss_beta_0.925':WeightedBCELoss(beta=0.925), 
        'WeightedBCELoss_beta_0.95':WeightedBCELoss(beta=0.95),
        'WeightedBCELoss':WeightedBCELoss(),
        'WeightedFocalLoss_beta_0.85':WeightedFocalLoss(beta=0.85),
        'WeightedFocalLoss_beta_0.875':WeightedFocalLoss(beta=0.875),
        'WeightedFocalLoss_beta_0.9':WeightedFocalLoss(beta=0.9), 
        'WeightedFocalLoss_beta_0.925':WeightedFocalLoss(beta=0.925),
        'WeightedFocalLoss_beta_0.95':WeightedFocalLoss(beta=0.95),
        'WeightedFocalLoss':WeightedFocalLoss(),
        'WeightedBCELoss_beta_0.999':WeightedBCELoss(beta=0.999)
    }

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

    # Select data depending on whether or not we use cross validation
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
    if precompute: # load precomputed spectrograms
        if augs != '':  # load an augmented dataset
            AUGMENT_PATH  = cfg("AUGMENT_PATH")
            train_data = AugmentDataset(df_train, SPEC_PATH, AUGMENT_PATH, augmentations = [augs], mode='train', labels=birds, augment_prob=aug_prob)
            val_data = SpecDataset(df_val, SPEC_PATH, mode='train', labels=birds) 
        else: 
            train_data = SpecDataset(df_train, SPEC_PATH, mode='train', labels=birds)
            val_data = SpecDataset(df_val, SPEC_PATH, mode='train', labels=birds)  

    else: # load raw audio; does not currently support augmentations due to computational considerations
        if augs != '':
            print('\n Warning: Augmentations cannot be applied if precompute is False \n')

        train_data = SimpleDataset(df_train, DATA_PATH, mode='train', labels=birds)
        val_data = SimpleDataset(df_val, DATA_PATH, mode='train', labels=birds)

    # The Selector class defines how to pick the data to train on from a recording
    train_selector = Selector(duration=max_duration, offset=offset_train, device='cpu') 

    train_loader = DataLoader(
        train_data, 
        batch_size=bs_train, 
        num_workers=8, 
        collate_fn=lambda x: collate_fn(x, load_all=False, sr=sr, duration=max_duration, selector=train_selector), # defined in train_utils.py
        shuffle=True, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_data, 
        batch_size=bs_val, 
        num_workers=8, 
        collate_fn=lambda x: collate_fn(x, load_all=True), # defined in train_utils.py
        shuffle=False, 
        pin_memory=True
    )


    # create model
    # the check_args function filters the chosen modules and keeps those deemed 
    # neccessary (defined using the keep keyword)
    transforms1_train = [ 
            SelectSplitData(duration, n_splits, offset=offset_train, sr=sr)
            # add more transforms here if desired
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

    transforms2 = [ 
        InstanceNorm()
    ]

    transforms2 = TransformApplier(
        check_args(transforms2, args)
    )

    # If we don't precompute the spectrograms, this module does so during runtime
    wav2spec = nn.Identity() if precompute else Wav2Spec()

    # we can now define the data pipelines
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
        in_chans=1, # normally 3 for RGB-images, but timm supports in_chans=1, appropriate here
    )

    # Post-CNN processing
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

    # metrics to report
    metric_f1micro = MultilabelF1Score(
        num_labels = num_classes, 
        topk = 1, # this means we say that we take the label with the highest probability for prediction
        average='micro', 
    ).to(device) 

    metric_f1_ours = F1Macro() 
    metric_recall_ours = RecallMacro() 
    metric_prec_ours = PrecisionMacro() 

    metrics = {
                'F1Micro': metric_f1micro,
                'F1Ours': metric_f1_ours,
                'RecallOurs': metric_recall_ours,
                'PrecisionOurs': metric_prec_ours,
            }
    # convert from a dict to our Metric class, used in the reporting implementation
    metrics = [
        Metric(name, method) for name, method in metrics.items()
    ]

    # We regularly save the models for (potential) later use using this class
    model_saver = ModelSaver(OUTPUT_DIR, experiment_name)

    # Next we define how to validate the data
    # The first step is to define the available policies
    policies = {
        'first_and_final': first_and_final,
        'max_thresh': max_thresh, 
        'max_all': max_all, 
        'rolling_avg': rolling_avg,     
    }
    policy = policies[args.policy]

    # The Validator class takes care of validating the data
    # by splitting long recordings into segments and returning
    # the final predictions for the full recording to be validated
    validator = Validator(
        data_pipeline_val, 
        model, 
        overlap=overlap, 
        device=device, 
        policy=policy,
        scheme=args.scheme
    )

    config = vars(args) # configuration reported to wandb

    # finally we define the trainer, which trains the model using the variables
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