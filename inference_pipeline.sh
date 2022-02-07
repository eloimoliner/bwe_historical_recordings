#!/bin/bash

module load anaconda 
source activate /scratch/work/molinee2/conda_envs/bwe_test
export TORCH_USE_RTLD_GLOBAL=YES


#PATH_CKPT=experiments_bwe/piano/checkpoint_149
#PATH_CKPT=experiments_bwe/strings/checkpoint_119
PATH_CKPT=experiments_bwe/orchestra/checkpoint_109
PATH_CKPT_DEN=experiments_denoiser/pretrained_model/checkpoint_29

name=$1

#AUDIO_EXAMPLES
#audio=/scratch/work/molinee2/datasets/real_noisy_data_test/examples_BWE/piano/ETUDE_IN_C-MOLL_Revolutions-Etude_-_Ignace_Jan_Paderewski_denoised.wav
#audio=/scratch/work/molinee2/datasets/real_noisy_data_test/orchestral/78_1st-movement-allegro-moderato-1st-record_philadelphia-symphony-orchestra-schubert_gbia7003512a/1st_Movement-Allegro_mod_-_PHILADELPHIA_SYMPHONY_ORCHESTRA_denoised.wav
audio=audio_examples/1st_Movement-Allegro_mod_-_PHILADELPHIA_SYMPHONY_ORCHESTRA_noisy_input.wav
#audio=/scratch/work/molinee2/datasets/real_noisy_data_test/demo_denoiser_torch/1st_Movement-Allegro_mod_-_PHILADELPHIA_SYMPHONY_ORCHESTRA_noisy_input__denoised.wav
#audio=/scratch/work/molinee2/datasets/real_noisy_data_test/orchestral/78_blue-danube_philadelphia-symphony-orchestra-johann-strauss-leopold-stokowski_gbia7003487a/BLUE_DANUBE_-_PHILADELPHIA_SYMPHONY_ORCHESTRA_denoised.wav

python inference_pipeline.py checkpoint=${PATH_CKPT} checkpoint_denoiser=${PATH_CKPT_DEN}  inference.audio=$audio $iteration  inference.apply_lpf=False   inference.use_denoiser=True inference.use_bwe=True inference.exp_name=$name
