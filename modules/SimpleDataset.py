import torch 
from torch.utils.data import DataLoader, Dataset
import librosa 
import numpy as np 
import pandas as pd

class SimpleDataset(Dataset):
    """
    Read the files from the specified paths and return a tensor representing the audio
    """
    def __init__(self, df, data_path, mode='train', labels=None) -> None:
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

    def setup_dataset(self, n_classes):
        """
        Convert dataframe entries to one-hot encodings of the classes, 
        """
        def one_hot(labels):
            """
            Return a list of one-hot encodings for passed labels
            labels: list
            """
            num2vec = lambda lbl: (np.arange(n_classes) == lbl).astype(int)
            if isinstance(labels[0], list):
                return [np.sum([num2vec(l) for l in lbls], axis=0) for lbls in labels]
            return [num2vec(lbl) for lbl in labels]
        
        def text_to_num(s: pd.Series):
            # if s is in self.labels, convert to that index, otherwise, set top index

            # check if this is the secondary label:
            bird2id = {bird: idx for idx, bird in enumerate(self.labels)}
            other = len(self.labels) # map non-existing entries to index len(labels)

            if '[' in s[0]:
                x = secondary_to_list(s)
                present = []
                for lbls in x:
                    present.append([bird2id[i] if i in self.labels else other for i in lbls])
                return present 
            
            return s.apply(lambda x: bird2id[x] if x in self.labels else other)

        def secondary_to_list(s: pd.Series):
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