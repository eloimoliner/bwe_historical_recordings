#!/bin/bash

module load anaconda 
source activate /scratch/work/molinee2/conda_envs/bwe_test

#n=23
#iteration=`sed -n "${n} p" iteration_parameters.txt`      # Get n-th line (2-indexed) of the file
#PATH_EXPERIMENT=/scratch/work/molinee2/unet_dir/bwe_historical_recordings/experiments_denoisr/pretrained_model

#audio=/scratch/work/molinee2/datasets/real_noisy_data_test/demo_denoiser_torch/_02-SecondRhapsodie-Part1-SergeiRachmaninoff_noisy_input.wav_22k.wav
audio=/scratch/work/molinee2/datasets/real_noisy_data_test/demo_denoiser_torch/Livery_Stable_Blues_noisy_input.wav
#_02-SecondRhapsodie-Part1-SergeiRachmaninoff.flac
#audio=/scratch/work/molinee2/datasets/real_noisy_data_test/demos/O_SOLE_MIO_ENRICO_CARUSO.flac
 
#audio=/scratch/work/molinee2/datasets/real_noisy_data_test/demos/Boulanger-Marsch.wav

python inference_denoiser.py inference.audio=$audio 

#path_experiment=${PATH_EXPERIMENT}  #checkpoint="checkpoint_24"

