#!/bin/bash

##SBATCH  --time=00:59:59
#SBATCH  --time=2-23:59:59
##SBATCH  --time=03:59:59
#SBATCH --mem=10G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=bwe_progressive_half_size
##SBATCH  --gres=gpu:a100:1
##SBATCH  --gres=gpu:1 --constraint=volta
##SBATCH  --gres=gpu:1 
#SBATCH  --gres=gpu:1 --exclude=gpu[20-27]
##SBATCH  --gres=gpu:1 --exclude=gpu[20-27]
##SBATCH  --gres=gpu:4 
##SBATCH  --gres=gpu:3 --constraint=volta
##SBATCH --output=/scratch/work/%u/unet_dir/unet_historical_music/experiments/23062021_metrics_mae/training_mse%j.out
#SBATCH --output=/scratch/work/%u/unet_dir/bwe_historical_recordings/experiments/%a_training_%j.out
#SBATCH --array=[41]

# ..mem=80G
module load anaconda 
source activate /scratch/work/molinee2/conda_envs/bwe_test
export TORCH_USE_RTLD_GLOBAL=YES

n=1
#n=2
iteration=`sed -n "${n} p" iteration_parameters.txt`      # Get n-th line (2-indexed) of the file

PATH_EXPERIMENT=/scratch/work/molinee2/unet_dir/bwe_historical_recordings/experiments/${n}
mkdir $PATH_EXPERIMENT


python train.py path_experiment="$PATH_EXPERIMENT"  $iteration 
#python train_NLD.py path_experiment="$PATH_EXPERIMENT"  $iteration 
