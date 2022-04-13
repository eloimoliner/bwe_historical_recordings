# BEHM-GAN: Bandwidth Extension of Historical Music using Generative Adversarial Networks

Official repository of the paper:

> E. Moliner and V. Välimäki,, "BEHM-GAN: Bandwidth Extension of Historical Music using Generative Adversarial Networks
", submitted to IEEE Transactions on Audio, Speech, and Language Processing, 2022

This repository also includes a PyTorch implementation of a denoiser based on the paper:

> E. Moliner and V. Välimäki,, "A two-stage U-Net for high-fidelity denosing of historical recordings", submitted to IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), Singapore, May, 2022


## Abstract
Audio bandwidth extension aims to expand the spectrum of narrow-band audio signals. Although this topic has been broadly studied during recent years, the particular problem of extending the bandwidth of historical music recordings remains an open challenge. This paper proposes BEHM-GAN, a model based on generative adversarial networks, as a practical solution to this problem. The proposed method works with the complex spectrogram representation of audio and, thanks to a dedicated regularization strategy, can effectively extend the bandwidth of out-of-distribution real historical recordings. The BEHM-GAN is designed to be applied as a second step after denoising the recording to suppress any additive disturbances, such as clicks and background noise. We train and evaluate the method using solo piano classical music. The proposed method outperforms the compared baselines in both objective and subjective experiments. The results of a formal blind listening test show that BEHM-GAN significantly increases the perceptual sound quality in early-20th-century gramophone recordings. For several items, there is a substantial improvement in the mean opinion score after enhancing historical recordings with the proposed bandwidth-extension algorithm. This study represents a relevant step toward data-driven music restoration in real-world scenarios. 

<p align="center">
<img src="https://user-images.githubusercontent.com/64018465/163122490-55aedb3b-7b21-46fc-838a-9fe90eb09b3e.png" alt="Training framework"
width="700px"></p>

Listen to our [audio samples](http://research.spa.aalto.fi/publications/papers/ieee-taslp-behm-gan/)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eloimoliner/bwe_historical_recordings/blob/main/colab/demo.ipynb)

## Requirements
You will need at least python 3.8 and CUDA 11.3 if you want to use GPU. See `environment.yaml` for the required package versions.

To install the environment through anaconda, follow the instructions:

    conda env update -f environment.yml
    conda activate historical_bwe_pytorch

You can enhance your recordings in the cloud using the Colab notebook. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eloimoliner/bwe_historical_recordings/blob/main/colab/demo.ipynb)

Otherwise, run the following commands to clone the repository and install the pretrained weights of the two-stage U-Net model:

    git clone https://github.com/eloimoliner/bwe_historical_recordings.git
    cd bwe_historical_recordings
    bash prepare_data.sh
    
## Inference

This repository includes a denoiser and a bandwidth extension models separatedly.

### Denoiser

    bash inference_pipeline.sh "file name"
    
### Denoiser + BWE pipeline

    bash inference_pipeline.sh "file name"
    
A ".wav" file with the denoised version, as well as the residual noise and the original signal in "mono", will be generated in the same directory as the input file.
## Training
TODO
## Remarks
The models included in this repository work at the reduced sampling rate of 22050Hz. 

The trained models are trained to extend the bandwidth above a cut-off frequency of about 3 kHz. So, the model is expected to perform better if the bandwidth of your music is in that range, which is the average in gramophone recordings.



