## TODO: take into in account the selectsplitdata for the audio files and compute the melspectrogram and save it



from PIL import Image
import PIL
import os
import torch
from os import listdir
from os.path import isfile, join, basename
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
transforms1 = TransformApplier(
    [
        SelectSplitData(duration=30, n_splits=6, offset=None),
        # add more transforms here
    ]
)
wav2img = nn.Sequential(transforms1,mel_spec, amplitude_to_db)
all_files = [os.path.join(path, name) for path, subdirs, files in os.walk(audio_path) for name in files]
for f in all_files:
    new_dir = SPEC_PATH+f[86:-12]
    try:
        os.makedirs(new_dir)
    except:
        pass
    wav, sr = librosa.load(f, sr=None, offset=0, duration=None)
    len = wav.shape[0]#[d[0].shape[-1] for d in data]
    img = wav2img(torch.Tensor(wav))
    torch.save(img,new_dir + f[-12:-4]+'.pt')

## Delete empty folders
root = SPEC_PATH
folders = list(os.walk(root))[1:]

for folder in folders:
    # folder example: ('FOLDER/3', [], ['file'])
    if not folder[2]:
        os.rmdir(folder[0])