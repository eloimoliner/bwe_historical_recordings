#!/bin/bash
#Downlad audio examples
wget https://github.com/eloimoliner/bwe_historical_recordings/releases/download/v0.0-alpha/audio_examples.zip
unzip audio_examples.zip -d audio_examples

#Download checkpoints BEHM-GAN
wget https://github.com/eloimoliner/bwe_historical_recordings/releases/download/v0.0-alpha/checkpoint_piano
mv checkpoint_piano  experiments_bwe/piano
wget https://github.com/eloimoliner/bwe_historical_recordings/releases/download/v0.0-alpha/checkpoint_strings
mv checkpoint_strings  experiments_bwe/strings
wget https://github.com/eloimoliner/bwe_historical_recordings/releases/download/v0.0-alpha/checkpoint_orchestra
mv checkpoint_orchestra  experiments_bwe/orchestra

#Download checkpoint denoiser
wget https://github.com/eloimoliner/bwe_historical_recordings/releases/download/v0.0-alpha/checkpoint_denoiser
mv checkpoint_denoiser  experiments_denoiser/pretrained_model
