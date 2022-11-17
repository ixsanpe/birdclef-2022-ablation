import torch 
from .SimpleDataset import SimpleDataset

class SpecDataset(SimpleDataset):
    """
    Initialized in the same way as SimpleDataset
    Only overwrites the __getitem__ method to load
    a spectrogram from memory instead of a wav file
    """
    def __getitem__(self, idx, debug=False):
        """
        Load a file and take the union of the primary and secondary labels as a label
        """
        path = f"""{self.data_path}{
            self.df.loc[idx, 'filename'].replace('.ogg', '.pt')
        }"""
        label = self.get_label(idx)
        
        if debug: print(path)
        
        spectrogram = torch.load(path)

        return spectrogram, label, path