# Modular layout

Looking at other submissions, they typically use a *Dataloading* and *Model* module, among which they split different tasks such as pre- and post-processing. Here, we make the layout even more modular in order to easily adapt which model we use. 

## Pipeline
- Data loading
    - Only loads data in wav format or as a precomputed spectrogram

- Pre-processing 1
    - Add metadata to each file (e.g. weight based on rating)
    - Pre-processing of the wav files themselves

- Transform to Mel Spectrogram if appropriate
    - Make a separate module for transforming to spectrograms, which most submissions in BirdClef train on

- Pre-processing 2
    - We might want to pre-process the spectrograms as well, before passing them to the model

- Model
    - Pass the modified spectrograms to a pre-trained model

- Post-processing
    - Apply post-processing to predictions
