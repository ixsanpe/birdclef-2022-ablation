import torch 
from torch.utils.data import DataLoader, Dataset
import librosa 
import numpy as np 
import pandas as pd

class SimpleDataset(Dataset):
    def __init__(
        self, 
        df: pd.DataFrame, 
        data_path, 
        mode: str='train', 
        labels: list=None
    ):
        """
        Read the files from the specified paths and return a tensor representing the audio
        Parameters:
            df: 
                dataframe containing information on training data file system 
                (at least 'filename', 'primary_label', 'secondary_label')
            data_path:
                path under which the data is stored in df['filename']
            mode:
                'train', 'validation' or 'test'; which mode to use this SimpleDataset
            labels:
                list of labels to be used with this data
        """
        super().__init__()
        self.df = df 
        self.data_path = data_path
        self.paths = df['filename']
        self.mode = mode 
        self.labels = labels 
        n_classes = len(labels)
        self.setup_dataset(n_classes)

    def __len__(self):
        return len(self.df)

    def setup_dataset(self, n_classes: int):
        """
        Convert dataframe entries to one-hot encodings of the classes, 
        Parameters:
            n_classes:
                The number of classes to one-hot encode
        """
        def one_hot(labels: list):
            """
            Return a list of one-hot encodings for passed labels
            Parameters
                labels: 
                    list of labels to one-hot encode
            Returns:
                one-hot encoded labels
            """
            num2vec = lambda lbl: (np.arange(n_classes) == lbl).astype(int)
            if isinstance(labels[0], list):
                return [np.sum([num2vec(l) for l in lbls], axis=0) for lbls in labels]
            return [num2vec(lbl) for lbl in labels]
        
        def text_to_num(s):
            """
            if s is in self.labels, convert to that index, otherwise, set top index
            Parameters:
                s:
                    iterable of str class names to be converted to indices
            Returns:
                the same datatype as s but with indices where the entries in s matched
                labels for this dataset. Otherwise, the index is len(self.labels). 
            """

            # check if this is the secondary label:
            bird2id = {bird: idx for idx, bird in enumerate(self.labels)}
            other = len(self.labels) # map non-existing entries to index len(labels)

            if '[' in s[0]: # list of lists
                x = secondary_to_list(s)
                present = []
                for lbls in x:
                    present.append([bird2id[i] if i in self.labels else other for i in lbls])
                return present 
            
            return s.apply(lambda x: bird2id[x] if x in self.labels else other)

        def secondary_to_list(s: pd.Series):
            """
            Given a str representation of a list (secondary labels), convert this to an actual list
            Parameters:
                s:
                    pd.Series with str entries encoding a list as "'[x_0, ..., x_n]'"
            Returns:
                the encoded list as a python list
            """
            x = s.copy()
            x = x.apply(lambda z: z.replace('[', ''))
            x = x.apply(lambda z: z.replace(']', ''))
            retval = [i.replace(' ', '').replace('\'', '').split(',') for i in x]
            return retval

        df = self.df.copy()
        
        if self.mode == 'train':
            primary_label = text_to_num(df['primary_label'])
            self.primary_label = one_hot(primary_label.astype(int).values)
            secondary_label = text_to_num(df['secondary_labels'])
            self.secondary_label = one_hot(secondary_label)
        
        else:
            raise NotImplementedError

    def __getitem__(self, idx, debug=False):
        if self.mode == 'train':
            path = f"train_audio/{self.df.loc[idx, 'filename']}"
            label = self.primary_label[idx] + self.secondary_label[idx]
            if debug:
                print(path)
        else:
            raise NotImplementedError

        duration = 5 # TODO
        offset = 0 # TODO
        wav = self.load_one(path, offset, duration)
        wav_tensor = torch.tensor(wav)

        return wav_tensor, label 

    def load_one(self, id_, offset, duration):
        fp = self.data_path + id_
        try:
            wav, sr = librosa.load(fp, sr=None, offset=offset, duration=duration)
        except:
            print("FAIL READING rec", fp)

        return wav