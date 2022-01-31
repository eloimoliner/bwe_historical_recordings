import os
import hydra
import logging

logger = logging.getLogger(__name__)

def run(args):
    import models.denoiser as denoiser
    import torch
    import soundfile as sf
    import numpy as np
    from tqdm import tqdm
    import utils
    from utils import do_stft
    import scipy.signal
    
    path_experiment=str(args.path_experiment_denoiser)
    checkpoint_filepath=os.path.join(path_experiment, args.checkpoint_denoiser)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet_model = denoiser.MultiStage_denoise(unet_args=args.denoiser)

    unet_model.load_state_dict(torch.load(checkpoint_filepath, map_location=device))
    #    
    unet_model.to(device)

    #ckpt=os.path.join(os.path.dirname(os.path.abspath(__file__)),path_experiment, 'checkpoint')
    #unet_model.load_weights(ckpt)

    #def do_stft(noisy):
    #    
    #    #window_fn = tf.signal.hamming_window

    ##    win_size=args.stft.win_size
    #    hop_size=args.stft.hop_size
    #    window=torch.hamming_window(window_length=win_size)
    #window=window.to(device)
    #    noisy=torch.cat((noisy, torch.zeros(1,win_size).to(device)), 1)
    #    stft_signal_noisy=torch.stft(noisy, win_size, hop_length=hop_size,window=window,center=False,return_complex=False)
    #    stft_signal_noisy=stft_signal_noisy.permute(0,3,2,1)
   
    #    return stft_signal_noisy

    #def do_istft(data):
    #    
    #    win_size=args.stft.win_size
    #    hop_size=args.stft.hop_size
    #    window=torch.hamming_window(window_length=win_size)
    #    window=window.to(device)
    #    data=data.permute(0,3,2,1)
    #    pred_time=torch.istft(data, win_size, hop_length=hop_size,  window=window, center=False, return_complex=False)

    #    return pred_time
    def apply_denoiser_model(segment):
        segment_TF=do_stft(segment,win_size=args.stft.win_size, hop_size=args.stft.hop_size, device=device)
        #segment_TF_ds=tf.data.Dataset.from_tensors(segment_TF)
        with torch.no_grad():
            pred = unet_model(segment_TF)
        if args.denoiser.num_stages>1:
            pred=pred[0]

        pred_time=utils.do_istft(pred, args.stft.win_size, args.stft.hop_size,device)
        pred_time=pred_time[0].detach().cpu().numpy()
        return pred_time


    audio=str(args.inference.audio)
    data, samplerate = sf.read(audio)
    #Stereo to mono
    if len(data.shape)>1:
        data=np.mean(data,axis=1)
    
    if samplerate!=22050: 
        print("Resampling")
   
        data=scipy.signal.resample(data, int((22050  / samplerate )*len(data))+1)  
 
    
    
    segment_size=22050*5  #20s segments

    length_data=len(data)
    overlapsize=1024 #samples (46 ms)
    window=np.hanning(2*overlapsize)
    window_right=window[overlapsize::]
    window_left=window[0:overlapsize]
    audio_finished=False
    pointer=0
    denoised_data=np.zeros(shape=(len(data),))
    residual_noise=np.zeros(shape=(len(data),))
    numchunks=int(np.ceil(length_data/segment_size))
     
    for i in tqdm(range(numchunks)):
        if pointer+segment_size<length_data:
            segment=data[pointer:pointer+segment_size]
            #dostft
            segment = torch.from_numpy(segment)
            segment=segment.type(torch.FloatTensor)
            segment=segment.to(device)
            segment=torch.unsqueeze(segment,0)
            pred_time=apply_denoiser_model(segment)

            if pointer==0:
                pred_time=np.concatenate((pred_time[0:int(segment_size-overlapsize)], np.multiply(pred_time[int(segment_size-overlapsize):segment_size],window_right)), axis=0)
            else:
                pred_time=np.concatenate((np.multiply(pred_time[0:int(overlapsize)], window_left), pred_time[int(overlapsize):int(segment_size-overlapsize)], np.multiply(pred_time[int(segment_size-overlapsize):int(segment_size)],window_right)), axis=0)
                
            denoised_data[pointer:pointer+segment_size]=denoised_data[pointer:pointer+segment_size]+pred_time

            pointer=pointer+segment_size-overlapsize
        else: 
            segment=data[pointer::]

            lensegment=len(segment)
            segment=np.concatenate((segment, np.zeros(shape=(int(segment_size-len(segment)),))), axis=0)

            audio_finished=True
            #dostft
            segment = torch.from_numpy(segment)
            segment=segment.type(torch.FloatTensor)
            segment=segment.to(device)
            segment=torch.unsqueeze(segment,0)
            pred_time=apply_denoiser_model(segment)

            if pointer==0:
                pred_time=pred_time
            else:
                pred_time=np.concatenate((np.multiply(pred_time[0:int(overlapsize)], window_left), pred_time[int(overlapsize):int(segment_size)]),axis=0)

            denoised_data[pointer::]=denoised_data[pointer::]+pred_time[0:lensegment]

    basename=os.path.splitext(audio)[0]
    wav_noisy_name=basename+"_noisy_input"+".wav"
    sf.write(wav_noisy_name, data, 22050)
    wav_output_name=basename+"_denoised"+".wav"
    sf.write(wav_output_name, denoised_data, 22050)
    

def _main(args):
    global __file__

    __file__ = hydra.utils.to_absolute_path(__file__)

    run(args)


@hydra.main(config_path="conf", config_name="conf")
def main(args):
    try:
        _main(args)
    except Exception:
        logger.exception("Some error happened")
        # Hydra intercepts exit code, fixed in beta but I could not get the beta to work
        os._exit(1)


if __name__ == "__main__":
    main()







