#!/bin/bash

##SBATCH  --time=00:59:59
#SBATCH  --time=2-23:59:59
#SBATCH --mem=10G
#SBATCH --cpus-per-task=10
#SBATCH --job-name=training_array_metrics
##SBATCH  --gres=gpu:a100:1
#SBATCH  --gres=gpu:1 --constraint=volta
##SBATCH  --gres=gpu:4 --constraint=volta
##SBATCH  --gres=gpu:3 --constraint=volta
##SBATCH --output=/scratch/work/%u/unet_dir/unet_historical_music/experiments/23062021_metrics_mae/training_mse%j.out
#SBATCH --output=/scratch/work/%u/unet_dir/denoising_pytorch/experiments/%a_training_%j.out
#SBATCH --array=[21,22,23]

# ..mem=80G
module load anaconda 
source activate /scratch/work/molinee2/conda_envs/2022_torchot

n=$SLURM_ARRAY_TASK_ID
n=3
iteration=`sed -n "${n} p" iteration_parameters_denoiser.txt`      # Get n-th line (2-indexed) of the file

PATH_EXPERIMENT=experiments_denoiser/${n}
mkdir $PATH_EXPERIMENT

python  train_denoiser.py path_experiment="$PATH_EXPERIMENT" $iteration  epochs=150  num_workers=10
