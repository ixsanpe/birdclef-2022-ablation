"""
Find the names of all the birds present in the dataset and save them in the
same format as scored_birds.json from the Kaggle challenge. 
"""

import pandas as pd 
import os
import json
import numpy as np
from decouple import config

def remove_chars(s, chars=['[', ']', ' ', '\'']):
    for c in chars:
        s = s.replace(c, '')
    return s

def extract_bird_names():
    DATA_PATH = config("DATA_PATH")
    OUTPUT_PATH = DATA_PATH
    df = pd.read_csv(f'{DATA_PATH}train_metadata.csv')
    primary_labels = df['primary_label'].replace('[', '').replace(']', '')
    primary_labels = pd.Series(pd.unique(primary_labels))
    secondary_labels = df['secondary_labels'].apply(lambda s: remove_chars(s).split(','))
    sec_labels = []
    for l in secondary_labels:
        sec_labels.extend(l)
    secondary_labels = pd.unique(pd.Series(sec_labels))
    labels = np.unique(np.concatenate([primary_labels, secondary_labels]))
    labels = np.delete(labels, np.argwhere(labels == ''))

    # save files in same format as 'scored_birds.json'
    labels = json.dumps(list(labels))
    with open(f"{OUTPUT_PATH}all_birds.json", "w") as f:
        f.write(labels)

    

if __name__ == '__main__':
    extract_bird_names()