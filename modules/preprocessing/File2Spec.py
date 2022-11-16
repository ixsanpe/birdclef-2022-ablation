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
    Transform an audio (wav) signal to a Mel spectrogram. Using the settings of Henkel et. al. as a default

    Code heavily based on Henkel et. al.
    https://github.com/ChristofHenkel/kaggle-birdclef2021-2nd-place/tree/26438069466242e9154aacb9818926dba7ddc7f0
    """

    #def __init__(self):

    def forward(self, d: dict):
        name_file = d['files'] #folder/nameOfFile (without .ogg)
        #file_path = os.path.join(SPEC_PATH, name_file)
        file_path = [os.path.join(SPEC_PATH, f+ '.pt') for f in name_file]
        #file_path = file_path + '.pt'
        d['x'] = torch.load(file_path) #list of tensors
        return d

