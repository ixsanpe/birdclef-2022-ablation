# %%
import pandas as pd
from skmultilearn.model_selection import iterative_train_test_split
import json
import numpy as np
import matplotlib.pyplot as plt

from modules import * 

from decouple import config

DATA_PATH = config("DATA_PATH")
SPEC_PATH = config('SPEC_PATH')
SPLIT_PATH = config('SPLIT_PATH')
AUGMENT_PATH = config('AUGMENT_PATH')

# %%
metadata_path = f'{DATA_PATH}train_metadata.csv'
bird_path = f'{DATA_PATH}all_birds.json'

# %%
# Load data 
with open(bird_path) as f:
    birds = json.load(f)

num_classes = len(birds)

metadata = pd.read_csv(metadata_path)
# %%

# Loads data into Dataset and extracts indices and labels from that
data = SimpleDataset(metadata, 'birdclef-2022/', mode='train', labels=birds)
y = np.logical_or(data.primary_label, data.secondary_label)
X = data.df.index.to_numpy().reshape(-1,1)

# %%
test_split = 0.05
val_slit = 0.05
augment_split = 0.5
random_state=42

# Uses the scikit-multilearn library to do a multilabel stratified fold
_,_, X_augment, y_augment = iterative_train_test_split(X, y, test_size = augment_split)
X_train, y_train, X_test_val, y_test_val = iterative_train_test_split(X, y, test_size = test_split + val_slit)
X_val, y_val, X_test, y_test = iterative_train_test_split(X_test_val, y_test_val, test_size = test_split/(val_slit+test_split))


# %%
# creates new metadata files and saves them
metadata_train = metadata.iloc[X_train.squeeze()]
metadata_val = metadata.iloc[X_val.squeeze()]
metadata_test = metadata.iloc[X_test.squeeze()]
metadata_augment = metadata.iloc[X_augment.squeeze()]

#metadata_train.to_csv(f'{SPLIT_PATH}train_metadata.csv')
#metadata_val.to_csv(f'{SPLIT_PATH}val_metadata.csv')
#metadata_test.to_csv(f'{SPLIT_PATH}test_metadata.csv')
#metadata_augment.to_csv(f'augment_metadata.csv')



# %%
augs = 'gain'
aug_prob = 0.5
spec_data = SpecDataset(metadata, SPEC_PATH, mode='train', labels=birds)
aug_data = AugmentDataset(metadata, SPEC_PATH, AUGMENT_PATH, augmentations = [augs], mode='train', labels=birds, augment_prob=aug_prob)
# %%
