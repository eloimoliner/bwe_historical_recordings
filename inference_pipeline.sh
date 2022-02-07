#!/bin/bash

module load anaconda 
source activate /scratch/work/molinee2/conda_envs/bwe_test


PATH_CKPT=experiments_bwe/piano/checkpoint_piano
#PATH_CKPT=experiments_bwe/strings/checkpoint_strings
#PATH_CKPT=experiments_bwe/orchestra/checkpoint_orchestra

PATH_CKPT_DEN=experiments_denoiser/pretrained_model/checkpoint_denoiser

#Some name to define the experiment
name=$1

#AUDIO_EXAMPLES
#PIANO
audio=audio_examples/HUNGARIAN_RHAPSODY_No._8_-_MARK_HAMBOURG_noisy_input.wav
#STRINGS
#audio=audio_examples/HUMORESQUE_-_VENETIAN_TRIO_-_DVOK_noisy_input.wav
#ORCHESTRA
#audio=audio_examples/1st_Movement-Allegro_mod_-_PHILADELPHIA_SYMPHONY_ORCHESTRA_noisy_input.wav

python inference_pipeline.py checkpoint=${PATH_CKPT} checkpoint_denoiser=${PATH_CKPT_DEN}  inference.audio=$audio $iteration  inference.apply_lpf=False   inference.use_denoiser=True inference.use_bwe=True inference.exp_name=$name
