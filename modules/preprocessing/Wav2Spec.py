import torch.nn as nn 
import torchaudio as ta 

class Wav2Spec(nn.Module):
    """
    Transform an audio (wav) signal to a Mel spectrogram. Using the settings of Henkel et. al. as a default

    Code heavily based on Henkel et. al.
    https://github.com/ChristofHenkel/kaggle-birdclef2021-2nd-place/tree/26438069466242e9154aacb9818926dba7ddc7f0
    """
    def __init__(
        self, 
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
        top_db=None, 
    ):
        super().__init__()
        self.mel_spec = ta.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                f_min=f_min,
                f_max=f_max,
                pad=pad,
                n_mels=n_mels,
                power=power,
                normalized=normalized,
            )

        self.amplitude_to_db = ta.transforms.AmplitudeToDB(top_db=top_db)
        self.wav2img = nn.Sequential(self.mel_spec, self.amplitude_to_db)
    
    def forward(self, d: dict):
        to_transform = d['x']
        transformed = self.wav2img(to_transform)
        d['x'] = transformed 
        return d 