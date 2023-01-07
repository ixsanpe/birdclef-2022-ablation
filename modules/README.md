# Modular layout

Looking at other submissions, they typically use a *Dataloading* and *Model* module, among which they split different tasks such as pre- and post-processing. Here, we make the layout even more modular in order to easily adapt which model we use. 

## Pipeline
- Data loading
    - Only loads data in wav format or as a precomputed spectrogram

- Pre-processing 1
    - Pre-processing of the wav files themselves (if appropriate)

- Transform to Mel Spectrogram if appropriate
    - Make a separate module for transforming to spectrograms, which most submissions in BirdClef train on

- Pre-processing 2
    - We might want to pre-process the spectrograms as well, before passing them to the model

- Model
    - Pass the modified spectrograms to a pre-trained model

- Post-processing
    - Apply post-processing to predictions

## The different subdirectories
We have split this directory, modules, into multiple subdirectories according to the type of files they contain:
- data: Modules and scripts relevant for preparing and loading data
- modelling: module for the pretrained model and utilities
- postprocessing: modules that modify the output of the pretrained model, so that they can be used for prediction
- preprocessing: augmentations and modules which are used to select and process parts of the data during training
- training: losses, Trainer, Validator, logging and metrics. Trainer and Validator are classes used to train the model and compute predictions on which to validate. Logging contains modules used by the Trainer to track progress.
