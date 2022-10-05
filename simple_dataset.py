import torch 
from torch.utils.data import DataLoader, Dataset
import librosa 

class SimpleDataset(Dataset):
    """
    Read the files from the specified paths and return a tensor representing the audio
    """
    def __init__(self, df, data_path, mode='train') -> None:
        super().__init__()
        self.df = df 
        self.data_path = data_path
        self.paths = df['filename']
        self.mode = mode 

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.mode == 'train':
            path = f"train_audio/{self.df.loc[idx, 'filename']}"
            label = self.df.loc[idx, 'primary_label'] # as simple as possible... TODO
        
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