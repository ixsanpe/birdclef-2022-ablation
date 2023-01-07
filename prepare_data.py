"""
This script prepares the data, given that you have
1. downloaded the birdclef-2022 data and saved it to the DATA_PATH defined in .env
2. downloaded the freefield data and saved it under the NOISE_PATH defined in .env

Then, the script will

- extract all the birds as targets
- make train-test splits
- convert audio to spectrograms and save those
- apply augmentations
"""


