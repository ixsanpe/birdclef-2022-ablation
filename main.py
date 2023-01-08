"""
This script runs the complete pipeline after downloading the data and specifying the appropriate paths in the .env file (see README.md)

Note: The script has to do a lot of computation. We recommend running each of the below scripts individually in the interest of time.
"""
import subprocess

def main():
    subprocess.call(['python', 'prepare_data.py'])
    subprocess.call(['python', 'train_ablation_original.py'])
    subprocess.call(['python', 'train_ablation_new.py'])


if __name__ == '__main__':
    main()