#!/bin/bash

module load anaconda 
source activate /scratch/work/molinee2/conda_envs/bwe_test
export TORCH_USE_RTLD_GLOBAL=YES

n=1
iteration=`sed -n "${n} p" iteration_parameters.txt`      # Get n-th line (2-indexed) of the file
PATH_EXPERIMENT=/scratch/work/molinee2/unet_dir/bwe_github/experiments/piano

#audio=/scratch/work/molinee2/datasets/real_noisy_data_test/demos/_02-SecondRhapsodie-Part1-SergeiRachmaninoff_denoised_good.wav
#audio=\'/scratch/work/molinee2/datasets/real_noisy_data_test/lt_bwe/real/01_-_Valse_dAdieux_As-dur_A_flat_majo_-_Alexander_Brailowsky_denoised.wav\'
#audio=\'/scratch/work/molinee2/datasets/real_noisy_data_test/demos/Evening_Song_\(Chant_du_Soir\)_-_Victor_String_Ensemble_denoised.wav\'
name=$1
add_noise=True
fc=3000
#REAL
#audio=/scratch/work/molinee2/datasets/real_noisy_data_test/test_strings/Canzonetta_-_Victor_String_Quartet_-_V._Hollaender_denoised.wav
#audio=/scratch/work/molinee2/datasets/real_noisy_data_test/test_strings/HUMORESQUE_-_VENETIAN_TRIO_-_DVOK_denoised.wav
audio=/scratch/work/molinee2/datasets/real_noisy_data_test/test_strings/78_humoresque_mischa-elman-josef-bonime-dvorak_gbia0283464a/Humoresque_-_Mischa_Elman_-_Josef_Bonime_denoised.wav
#audio=/scratch/work/molinee2/datasets/real_noisy_data_test/orchestral/78_blue-danube_philadelphia-symphony-orchestra-johann-strauss-leopold-stokowski_gbia7003487a/BLUE_DANUBE_-_PHILADELPHIA_SYMPHONY_ORCHESTRA_denoised.wav
#audio=/scratch/work/molinee2/datasets/real_noisy_data_test/orchestral/78_1st-movement-allegro-moderato-1st-record_philadelphia-symphony-orchestra-schubert_gbia7003512a/1st_Movement-Allegro_mod_-_PHILADELPHIA_SYMPHONY_ORCHESTRA_denoised.wav
#audio=\'/scratch/work/molinee2/datasets/real_noisy_data_test/orchestral/78_carmen-fantasie-i-teil-fantasia-part-i-fantasia-parte-ia_the-opera-orches_gbia7003949a/Carmen_Fantasie,_I._Teil__-_The_Opera_Orchestra,_Berlin_denoised.wav\'
#audio=\'/scratch/work/molinee2/datasets/real_noisy_data_test/opera/Carmen-_Habanera_Love_is_Like_a_Wood_Bir_-_Sophie_Braslau_denoised_denoised.wav\'

#78_carmen-fantasie-i-teil-fantasia-part-i-fantasia-parte-ia_the-opera-orches_gbia7003949a/Carmen_Fantasie,_I._Teil__-_The_Opera_Orchestra,_Berlin_denoised.wav\'
#audio=/scratch/work/molinee2/datasets/real_noisy_data_test/test_strings/78_2nd-movement-largo-andante-concluded-3rd-mov-vivace_gregor-piatigorsky-the_gbia0253483b/2nd_Movement_-_Largo_Andante_-_concl_-_Gregor_Piatigorsky_denoised.wav
##audio=/scratch/work/molinee2/datasets/real_noisy_data_test/test_strings/Eleanor_-_Victor_String_Orchestra_-_Jessie_L._Deppen_denoised.wav
#audio=/scratch/work/molinee2/datasets/real_noisy_data_test/test_strings/Evening_Song_Chant_du_Soir_-_Victor_String_Ensemble_denoised.wav

python inference.py path_experiment=${PATH_EXPERIMENT}  inference.audio=$audio $iteration  checkpoint="checkpoint_149" inference.apply_lpf=False inference.exp_name=$name bwe.add_noise.add_noise=$add_noise inference.fc=$fc
