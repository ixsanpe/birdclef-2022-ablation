
import pandas as pd
from skmultilearn.model_selection import iterative_train_test_split, IterativeStratification
import json
import numpy as np
import os
from SimpleDataset import SimpleDataset
import argparse 
from decouple import config

DATA_PATH = config("DATA_PATH")
SPEC_PATH = config('SPEC_PATH')
SPLIT_PATH = config('SPLIT_PATH')
SPLIT_PATH_KFOLD = config('SPLIT_PATH_KFOLD')

def split_data(save_path, train_split = 0.9 , test_split = 0.05, val_split = 0.05):

    assert train_split + test_split + val_split == 1.0
    os.makedirs(save_path, exist_ok=True)
    metadata_path = f'{DATA_PATH}train_metadata.csv'
    bird_path = f'{DATA_PATH}all_birds.json'

    # Load data 
    with open(bird_path) as f:
        birds = json.load(f)

    num_classes = len(birds)

    metadata = pd.read_csv(metadata_path)

    # Loads data into Dataset and extracts indices and labels from that
    data = SimpleDataset(metadata, 'birdclef-2022/', mode='train', labels=birds)
    y = np.logical_or(data.primary_label, data.secondary_label)
    X = data.df.index.to_numpy().reshape(-1,1)


    # Uses the scikit-multilearn library to do a multilabel stratified fold
    X_train, y_train, X_test_val, y_test_val = iterative_train_test_split(X, y, test_size = test_split + val_split)
    X_val, y_val, X_test, y_test = iterative_train_test_split(X_test_val, y_test_val, test_size = test_split/(val_split+test_split))


    # creates new metadata files and saves them
    metadata_train = metadata.iloc[X_train.squeeze()]
    metadata_val = metadata.iloc[X_val.squeeze()]
    metadata_test = metadata.iloc[X_test.squeeze()]

    metadata_train.to_csv(os.path.join(save_path, 'train_metadata.csv'))
    metadata_val.to_csv(os.path.join(save_path, 'val_metadata.csv'))
    metadata_test.to_csv(os.path.join(save_path, 'test_metadata.csv'))


def k_split_data(save_path, k=3):
    metadata_path = f'{DATA_PATH}train_metadata.csv'
    bird_path = f'{DATA_PATH}all_birds.json'
    

    # Load data 
    with open(bird_path) as f:
        birds = json.load(f)

    metadata = pd.read_csv(metadata_path)

    # Loads data into Dataset and extracts indices and labels from that
    data = SimpleDataset(metadata, 'birdclef-2022/', mode='train', labels=birds)
    y = np.logical_or(data.primary_label, data.secondary_label) # labels
    X = data.df.index.to_numpy().reshape(-1,1) # indices in metadata.csv


    # Uses the scikit-multilearn library to do a multilabel stratified fold

    k_fold = IterativeStratification(n_splits=k, order=1)
    for i, (train, test) in enumerate(k_fold.split(X, y)):
        save_path_k = os.path.join(save_path, str(i))
        os.makedirs(save_path_k, exist_ok=True)

        X_train, y_train = X[train], y[train]
        X_test, y_test = X[test], y[test]

        metadata_train = metadata.iloc[X_train.squeeze()]
        metadata_test = metadata.iloc[X_test.squeeze()]

        metadata_train.to_csv(os.path.join(save_path_k, 'train_metadata.csv'))
        metadata_test.to_csv(os.path.join(save_path_k, 'val_metadata.csv'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', choices=[1, 3], type=int, help='number of splits for which to split the data')
    args = parser.parse_args()
    k = args.k

    print(f'splitting data with {k=}')
    if k == 1:
        save_path = SPLIT_PATH
        split_data(save_path, train_split=.95, test_split=0., val_split=.05)
    else:
        assert k==3, f'We have implemented this only for k=3 but got {k=}. See .env file'
        save_path = SPLIT_PATH_KFOLD
        k_split_data(save_path, k=k)
    

