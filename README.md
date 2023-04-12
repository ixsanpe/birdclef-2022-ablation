# birdclef-2022-ablation
## AI4Good project: Approach to the [Kaggle BirdClef2022](https://www.kaggle.com/c/birdclef-2022/data) Challenge. 
The project aims to design an ablation study to automatically include and exclude modules in the pipeline to estimate the influence of preprocessing, models, postprocessing steps. The code can be re-used for other projects!

#### Setup
- Create a conda environment with all dependencies:
`conda create -n ai4good -f environment.yml python=3.10`
`conda activate ai4good`

- The path to the data and the repository is read from environment variables which you have to create: 
- You will need `pip install python-decouple` (or requirements.txt, which does so automatically)
1. `touch .env`
2. `nano .env`
3. add the variables to indicate each directory:
The directory where we extract the zip file from [Kaggle BirdClef2022](https://www.kaggle.com/c/birdclef-2022/data)
* DATA_PATH=./birdclef-2022/data/

The directory where we save the configuration for training models 
* OUTPUT_DIR=./birdclef-2022/output/

The directory where we pre-save the spectrograms of the audio file to speed up training 
* SPEC_PATH=./birdclef-2022/data/spec/

The directory where the splits for the data are saved
* SPLIT_PATH=./birdclef-2022/splits/split_1/

The directory where we pre-save the spectrograms augmentations 
* AUGMENT_PATH = ./birdclef-2022/data/aug/

The directory where the splits for CV are done
* SPLIT_PATH_KFOLD=./birdclef-2022/splits/3_fold_split/

The directory where the data from freefield is stored
* NOISE_PATH=./freefield/freefield1010_nobirds/wav/

Make sure you have the rights to these directories since otherwise, writing to them is not possible.

Of course, you can also download the data and set the paths above as you wish.
If you make your own directories, make sure they exist since the code will stop e.g. if it cannot save a model. 
In that case, make sure that ``DATA_PATH`` contains what you download from [the BirdClef 2022 Challenge](https://www.kaggle.com/competitions/birdclef-2022/data), and ``NOISE_PATH`` contains the [freefield noise data](https://archive.org/details/freefield1010)

### Running the Code
WARNING: If you decide to recompute the data used for training, please use _your own directories_ in the .env file above.
The reason for this is that we have prepared working data under the listed files above, and do not want to replace it. 
If the data contained in the above files remain unchanged, it should still be possible to train new models, in case there is an unexpected issue with the data preparation pipeline.

To run the full pipeline, set up the .env file as described above and run `main.py`. This will however be *very slow*, and we recommend not recomputing the data, as we shall describe in the following.

The main scripts are `prepare_data.py`, `train.py`, `train_ablation_new.py`, and `train_ablation_original.py`.
`prepare_data.py` prepares all of the data and will take a lot of time to do so, because computing the audio augmentations is very costly. 
Instead, we recommend setting the paths as listed above and running the scripts containing "ablation", as these are the ones that generate the output relevant for the report.

`train.py` is a regular train script which can be used to train a regular model, or run a cross validation study with three splits.
It takes a number of arguments, a description of which can be found in `train.py`, and allows creating many different models using different pre- and post-processing steps, augmentations and (pretrained) model architectures.
On the other hand, the `train_ablation` scripts call `train.py` with different parameters, instantiating different pipelines in doing so.
`train_ablation_original.py` runs the ablation study for the original baseline, whereas `train_ablation_new.py` runs it for the changed baseline. 

### Troubleshooting
We have experienced our fair share of issues concerning folder permissions, missing files and corrupted data.
While we have of course done our best to avoid all of these mistakes, it is not possible to test everything, especially since some errors only occur sometimes.
Something that might help is re-running the program (using e.g. only one core), in order to avoid issues with multiprocessing. 
Please do not hesitate to reach out if you obtain any strange errors when running this code. 

### Comments on the repository
There are many folders in this repository. 
They are organized as follows:
- ablation: contains all the code which is relevant for calling `train.py` with different configurations sequentially
- modules: contains the bulk of the code in the project. All of the modules in the pipeline can be found here (see `modules/readme.md`).
- splits: contains the train and validation splits for each fold in the cross validation, and also the train and validation split when not using cross validation.
- output: saved models during runs. 
- data: all of the datasets and metadata (also augmented data and precomputed spectrograms).

#### Datasets

BirdClef 2022: https://www.kaggle.com/competitions/birdclef-2022/data

Freefield1010 Data (for augmentations): https://archive.org/details/freefield1010

