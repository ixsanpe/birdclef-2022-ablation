"""
This script prepares the data, given that you have
1. downloaded the birdclef-2022 data and saved it to the DATA_PATH defined in .env
2. downloaded the freefield data and saved it under the NOISE_PATH defined in .env

Then, the script will

- extract all the birds as targets
- make train-test splits
- convert audio to spectrograms and save those
- apply augmentations

Note that you can queue this script on a cluster without getting banned since
we are using the subprocess library. Therefore 'python', '{script}.py' does not
run directly on the cluster, but only as a subprocess of the initialized process,
therefore not exceeding any quota
"""

import subprocess 


def main():
    print('\nextracting bird names\n')
    subprocess.call(['python', 'extract_bird_names.py'])

    print('\nmaking data splits\n')
    subprocess.call(['python', 'modules/data/k_split_data.py', '--k', '1'])
    subprocess.call(['python', 'modules/data/k_split_data.py', '--k', '3'])

    print('\nextracting spectrograms\n')
    subprocess.call(['python', 'extract_melspectro.py'])

    print('\napplying augmentations\n')
    augmentations = [ 
        'gain',
        'gaussiannoise',
        'timestretch',
        'pitchshift',
        'shift',
        'backgroundnoise',
        'timemask',
        'frequencymask',
    ]
    for augmentation in augmentations:
        subprocess.call(['python', 'modules/preprocessing/compute_augmentations.py', augmentation])
    
    print('\nfinished\n')


if __name__ == '__main__':
    main()
