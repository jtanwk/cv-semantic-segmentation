#!/bin/bash
#
#SBATCH --output=/home/jonathantan/cv/run.stdout
#SBATCH --error=/home/jonathantan/cv/run.stderr
#SBATCH --workdir=/home/jonathantan/cv
#SBATCH --partition=titan
#SBATCH --job-name=jt_cv_project
#SBATCH --nodes=1
#SBATCH --mem=40000
#SBATCH --gres=gpu:1

python3 train_seg.py --load_saved_model True
