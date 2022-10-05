from typing import Callable
import torch 
import torch.nn as nn
import torchaudio as ta


class CNN(nn.Module):
    def __init__(
        self, 
        conv_depth: int, 
        kernel_sizes: list, 
        pool_kernel_sizes: list, 
        strides: list, 
        w_in: int, 
        h_in: int,
        n_channels: list,
        fc_dims: list, 
        paddings=None, 
        activation: Callable=nn.ReLU(), 
        pool: Callable=nn.MaxPool2d
    ):
        super().__init__()
        assert len(kernel_sizes) == conv_depth, 'different number of kernel sizes than conv filters!'
        assert len(n_channels) == conv_depth + 1, 'must have one more channel than depth due to final output'
        if paddings is None:
            paddings = [0]* conv_depth
        out_dim = lambda w, p, d, k, s: (w+2*p-d*(k-1)-1)//s+1 # d is dilation

        w_ins = [w_in]
        h_ins = [h_in]
        for i in range(conv_depth):
            w1 = out_dim(w_ins[i], paddings[i], 1, kernel_sizes[i], strides[i])
            w1 = out_dim(w1, 0, 1, pool_kernel_sizes[i], pool_kernel_sizes[i])
            w_ins.append(w1)
            h1 = out_dim(h_ins[i], paddings[i], 1, kernel_sizes[i], strides[i])
            h1 = out_dim(h1, 0, 1, pool_kernel_sizes[i], pool_kernel_sizes[i])
            h_ins.append(h1)
            
        convs = [ 
            nn.Conv2d(
                c_in, 
                c_out, 
                kernel_size=ks, 
                stride=st, 
                dilation=1, 
                padding=p
            )
            for c_in, c_out, ks, st, p in zip(n_channels[:-1], n_channels[1:], kernel_sizes, strides, paddings)
        ]
        self.convs = nn.ModuleList(convs)
        conv_out = w_ins[-1] * h_ins[-1] * n_channels[-1]
        fc_dims = [conv_out] + fc_dims
        fc = [
            nn.Linear(in_size, out_size) for in_size, out_size in zip(fc_dims[:-1], fc_dims[1:])
        ]
        self.fc = nn.ModuleList(fc)
        self.activation = activation
        self.pools = [pool(ks) for ks in pool_kernel_sizes]

    def forward(self, x):
        for p, c in zip(self.pools, self.convs):
            x = p(c(x))
        x = nn.Flatten()(x)
        for l in self.fc:
            x = self.activation(l(x))
        return x 

class Wav2Spec(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mel_spec = ta.transforms.MelSpectrogram(
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

        self.amplitude_to_db = ta.transforms.AmplitudeToDB(top_db=None)
        self.wav2img = torch.nn.Sequential(self.mel_spec, self.amplitude_to_db)
    def forward(self, x):
        return self.wav2img(x)

class WavImgCNN(nn.Module):
    def __init__(
        self, 
        conv_depth: int, 
        kernel_sizes: list, 
        pool_kernel_sizes: list, 
        strides: list, 
        w_in: int, 
        h_in: int,
        n_channels: list,
        fc_dims: list, 
        paddings=None, 
        activation: Callable=nn.ReLU(), 
        pool: Callable=nn.MaxPool2d
    ):
        super().__init__()
        self.cnn = CNN(
            conv_depth,
            kernel_sizes,
            pool_kernel_sizes,
            strides, 
            w_in, 
            h_in,
            n_channels,
            fc_dims,
            paddings=paddings,
            activation=activation,
            pool=pool,
        )
        self.wav2img = Wav2Spec()

    def forward(self, x):
        return self.cnn(self.wav2img(x))