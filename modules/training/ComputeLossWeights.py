#from ..data import SimpleDataset
from decouple import config
import numpy as np
import pandas as pd
import os
import json

DATA_PATH = config("DATA_PATH")
OUTPUT_DIR = config("OUTPUT_DIR")
with open(f'{DATA_PATH}all_birds.json') as f:
    birds = json.load(f)

class ComputeLossWeights():
    def __init__(self, beta=0.9,all_birds=birds,metadata=pd.read_csv(f'{DATA_PATH}train_metadata.csv')):
        super().__init__()
        self.beta = beta 
        self.birds = all_birds
        self.metadata = metadata



    def remove_chars(self, s, chars=['[', ']', ' ', '\'']):
        for c in chars:
            s = s.replace(c, '')
        return s

    def forward(self):
        # Get the primary and secondary labels from self.metadata

        df = self.metadata 
        primary_labels = df['primary_label'].replace('[', '').replace(']', '')
        primary_labels = pd.Series(primary_labels)
        secondary_labels = df['secondary_labels'].apply(lambda s: self.remove_chars(s).split(','))
        sec_labels = []
        for l in secondary_labels:
            sec_labels.extend(l)
        secondary_labels = pd.Series(secondary_labels)

        # Here we do not distinguish primary and secondary labels
        labels = np.concatenate([primary_labels, secondary_labels])
        labels = np.delete(labels, np.argwhere(labels == ''))

        counts=[0]*len(self.birds)
        i=0
        for bird in self.birds:
            counts[i]=max(sum(labels==bird),1) 
            i=i+1


        #Now, compute weights for the loss as described in https://towardsdatascience.com/handling-class-imbalanced-data-using-a-loss-specifically-made-for-it-6e58fd65ffab
        counts=np.array(counts)
        weights=(1-self.beta)/(1-self.beta**counts)
        weights1 = weights/max(weights) # normalize so that the maximum weight is always 1
        return weights1


