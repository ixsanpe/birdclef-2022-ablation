"""
Script for pre-computing the spectrograms and saving them under SPEC_PATH in .env
"""


import os
import torch
import torch.nn as nn
import torchaudio as ta
from decouple import config
import librosa
from modules import *
from modules.training.train_utils import *

SPEC_PATH = config("SPEC_PATH")
DATA_PATH = config("DATA_PATH")
audio_path = os.path.join(DATA_PATH, "train_audio")

mel_spec = ta.transforms.MelSpectrogram(
            sample_rate=32000,
            n_fft=1024,
            win_length=1024,
            hop_length=320,
            f_min=50,
            f_max=14000,
            pad=0,
            n_mels=64,
            power=2.0,
            normalized=False,
            )
amplitude_to_db = ta.transforms.AmplitudeToDB(top_db=None)

wav2img = nn.Sequential(
    mel_spec, 
    amplitude_to_db
)

all_files = [os.path.join(path, name) for path, subdirs, files in os.walk(audio_path) for name in files]
for f in all_files:
    bird_name, file_name = f.split('/')[-2:]
    new_dir = SPEC_PATH + bird_name + '/'
    target = new_dir + file_name.replace('.ogg', '.pt')
    if not os.path.isfile(target):
        os.makedirs(new_dir, exist_ok=True)
        wav, sr = librosa.load(f, sr=None, offset=0, duration=None)
        len = wav.shape[0] 
        img = wav2img(torch.Tensor(wav))
        torch.save(img, target)

## Delete empty folders
root = SPEC_PATH
folders = list(os.walk(root))[1:]

for folder in folders:
    # folder example: ('FOLDER/3', [], ['file'])
    if not folder[2]:
        os.rmdir(folder[0])