# birdclef-2022
AI4Good project


#### Key points for repo:
- The path to the data and the repository is now read from environment variables which you have to create: 
- You will need $ pip install python-decouple
1. $ touch .env 
2. $ nano .env 
3. add the variables DATA_PATH and OUTPUT_DIR, in Euler they are:
* DATA_PATH=/cluster/work/igp_psr/ai4good/group-2b/birdclef-2022/data/
* OUTPUT_DIR=/cluster/work/igp_psr/ai4good/group-2b/birdclef-2022/output/
* SPEC_PATH=/cluster/work/igp_psr/ai4good/group-2b/birdclef-2022/data/spec

##### References:

Some Links:

- https://www.researchgate.net/publication/362592701_Overview_of_BirdCLEF_2022_Endangered_bird_species_recognition_in_soundscape_recordings
- http://ceur-ws.org/Vol-3180/paper-170.pdf
- http://ceur-ws.org/Vol-2936/paper-134.pdf (2nd place 2021)


Dataset:
- https://www.kaggle.com/competitions/birdclef-2022/data

AUGMENTATION (Library audiomentations)
- https://github.com/iver56/audiomentations
