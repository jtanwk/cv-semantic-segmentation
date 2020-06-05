#!/bin/bash
#
#SBATCH --output=/home/jonathantan/cv/project/%j.stdout
#SBATCH --error=/home/jonathantan/cv/project/%j.stderr
#SBATCH --workdir=/home/jonathantan/cv/project
#SBATCH --partition=titan
#SBATCH --job-name=jt_cv_project
#SBATCH --nodes=1
#SBATCH --mem=8000
#SBATCH --gres=gpu:1

python3 train_cls.py
