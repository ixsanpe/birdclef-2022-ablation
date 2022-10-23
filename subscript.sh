#!/bin/bash
#SBATCH --output=sbatch_logs/%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
ls /vol/cuda/
source /scratch_net/biwidl311/lhauptmann/miniconda3/etc/profile.d/conda.sh
conda activate BirdClef
nvcc --version
python -u /scratch_net/biwidl311/lhauptmann/birdclef-2022/train.py