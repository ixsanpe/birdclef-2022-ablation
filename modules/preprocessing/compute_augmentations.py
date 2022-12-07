
import os
import torch
import torch.nn as nn
import torchaudio as ta
import audiomentations as am
from decouple import config
import librosa
from modules import *
from modules.training.train_utils import *
import json
import sys

DATA_PATH = config("DATA_PATH")
audio_path = os.path.join(DATA_PATH, "train_audio")
target_path = os.path.join(DATA_PATH, "aug")
noise_path = '/cluster/work/igp_psr/ai4good/group-2b/freefield/freefield1010_nobirds/wav'

'''
Predcompute augmentation file

Run in your console 
    python compute_augmentations.py AUGMENTATION

Options for AUGMENTATION
    - gain
    - gaussiannoise
    - timestrecht
    - pitchshift
    - shift
    - backgroundnoise

To add more options, add them into transformations dict in line 68


'''

sample_rate=32000

gain_params = {'min_gain_in_db' : -15.0,
                'max_gain_in_db': 5.0,
                'p' : 1.0}

gaussiannoise_params = {'min_amplitude': 0.001,
                        'max_amplitude': 0.015,
                        'p' : 1.0}

timestretch_params = {'min_rate': 0.8,
                        'max_rate':1.25,
                        'p': 1.0}

pitchshift_params = {'min_semitones': -4,
                        'max_semitones': 4,
                        'p': 1.0}

shift_params = {'min_fraction': -0.5,
                'max_fraction': 0.5,
                'p': 1.0}

timemask_params = {'min_band_part': 0.0,
                    'max_band_part': 0.5,
                    'fade': False,
                    'p': 1.0}
frequencymask_params = {'min_frequency_band': 0.03,
                        'max_frequency_band': 0.25,
                        'p': 1.0}

backgroundnoise_params = {'sounds_path': noise_path,
                        'min_snr_in_db': 3.0,
                        'max_snr_in_db': 30.0, 
                        'noise_transform': am.PolarityInversion(), 
                        'noise_rms': 'relative',
                        'p': 1.0}


transformations = {'gain': (am.Gain, gain_params),
                    'gaussiannoise': (am.AddGaussianNoise, gaussiannoise_params),
                    'timestretch': (am.TimeStretch, timestretch_params),
                    'pitchshift': (am.PitchShift, pitchshift_params),
                    'shift': (am.Shift, shift_params),
                    'backgroundnoise': (am.AddBackgroundNoise, backgroundnoise_params),
                    'timemask': (am.TimeMask, timemask_params),
                    'frequencymask': (am.FrequencyMask, frequencymask_params)}




def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def augment_files(transform, target_path, domain='wav', sample_rate=32000):
    mel_spec = ta.transforms.MelSpectrogram(
                sample_rate=sample_rate,
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
    for i, f in enumerate(all_files):
        print(f'Progress: {(i/len(all_files)*100):.1f}')
        bird_name, file_name = f.split('/')[-2:]
        new_dir = os.path.join(target_path ,bird_name)
        target = os.path.join(new_dir, file_name.replace('.ogg', '.pt'))
        if not os.path.isfile(target):
            os.makedirs(new_dir, exist_ok=True)
            wav, sr = librosa.load(f, sr=None, offset=0, duration=None)
            if domain=='wav':
                wav = transform(np.array(wav), sample_rate=sample_rate)
            img = wav2img(torch.from_numpy(wav))
            if domain=='spec':
                img = transform(np.array(img))
            torch.save(img, target)
            



def main(argv):
    
    key = argv[0]
    if len(argv) > 1:
        domain = argv[1]
    else:
        domain = 'wav'

    print(f"\n##################################\nComputing {key}\n##################################\n")

    
    (transormation, args) = transformations[key]

    save_path = os.path.join(target_path, f'aug_{key}')
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, 'param_dict.json'), 'w') as outfile:
        try:
            json.dump(args, outfile, skipkeys=True)
        except:
            json.dump({key: str(args[key]) for key in args.keys()}, outfile, skipkeys=True)

    augment_files(transormation(**args), save_path, domain=domain)


if __name__ == "__main__":
   main(sys.argv[1:])