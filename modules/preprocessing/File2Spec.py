import torch
from torch import nn
import torchaudio as ta
from PIL import Image
import PIL
import os
from decouple import config

SPEC_PATH = config("SPEC_PATH")

class File2Spec(nn.Module):
    """
    Load a precomputed spectrogram. In the forward call, provide a dictionary with the keys
    'files': from where do we load the data
    'x': the value associated with 'x' is loaded from the file
    """


    def forward(self, d: dict):
        """
        Load precomputed spectrograms from the files defined in d['files']
        This method overwrites d['x'] using the loaded values
        """
        name_file = d['files'] # folder/nameOfFile (without .ogg)
        file_path = [os.path.join(SPEC_PATH, f+ '.pt') for f in name_file]
        
        d['x'] = torch.stack([torch.load(f) for f in file_path])
        return d

