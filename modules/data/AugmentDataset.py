import torch 
from .SpecDataset import SimpleDataset
import pandas as pd
import os

class AugmentDataset(SimpleDataset):
    """
    Initialized in the same way as SimpleDataset
    Only overwrites the __getitem__ method to load
    a spectrogram from multiple dfs instead of one
    """

    def __init__(
        self, 
        df: pd.DataFrame, 
        data_path, 
        augment_path:str,
        augmentations:list,
        mode: str='train', 
        labels: list=None,
        augment_prob: float = 1.0
    ):

        """
        This class adds augmented recordings to the train
        augmentations : list of augmentations
                        possible choices:     
                                - gain
                                - gaussiannoise
                                - timestrecht
                                - pitchshift
                                - shift
                                - backgroundnoise
                for each augmentation, the size of the dataset is increase by  augment_prob*dataset_size
        augment_prob: The amount of augmented data in relation to the size of the normal dataset
        
        """
        super().__init__(df, data_path, mode, labels)
        self.augment_path = augment_path
        self.augment_prob = augment_prob
        self.df['aug'] = ''

        # Add extra elements to df, which contain a column that tells us the augmentation
        for aug in augmentations:
            aug_df = self.df.copy()
            aug_df['aug'] = aug
            if self.augment_prob < 1.0:
                aug_df = aug_df.sample(frac=self.augment_prob, replace=False)
            self.df = pd.concat((self.df, aug_df), ignore_index=True)

        # Need to rerun this after changing the df size
        n_classes = len(labels)
        self.setup_dataset(n_classes)

        if self.mode == 'test':
            print('\nWarning: Do you really want to have a validation set with Augmentations?')

    def __getitem__(self, idx, debug=False):

        """
        Load a file and take the union of the primary and secondary labels as a label
        """
        if self.df.loc[idx, 'aug'] != "": # Use spectrogram from pre-saved augmented data
            path = os.path.join(self.augment_path, 'aug_' + self.df.loc[idx, 'aug'], self.df.loc[idx, 'filename'].replace('.ogg', '.pt'))
            if not os.path.isfile(path):
                raise FileNotFoundError(f'Augmentation file {path} not found')
        else: # Normal case, use spectrogram
            path = f"""{self.data_path}{
                self.df.loc[idx, 'filename'].replace('.ogg', '.pt')
            }"""
        label = self.get_label(idx)
        
        if debug: print(path)
        
        spectrogram = torch.load(path)

        return spectrogram, label, path

