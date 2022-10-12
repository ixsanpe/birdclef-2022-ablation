"""
This script shows how to make a very simple model using the modules in this directory.
"""

from PretrainedModel import * 
from SimpleDataset import * 
from TransformApplier import * 
from Wav2Spec import *

def main():
    transforms1 = TransformApplier([nn.Identity()])

    wav2spec = Wav2Spec()

    transforms2 = TransformApplier([nn.Identity()])

    cnn = PretrainedModel(
        model_name='efficientnet_b2', 
        in_chans=1, # normally 3 for RGB-images
    )

    transforms3 = TransformApplier([nn.Identity()])

    output_head = OutputHead(n_in=cnn.get_out_dim(), n_out=21)

    model = nn.Sequential(
        transforms1, 
        wav2spec,
        transforms2, 
        cnn,
        transforms3, 
        output_head,
    )

if __name__ == '__main__':
    main()
