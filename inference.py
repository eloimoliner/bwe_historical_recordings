import os
import hydra
import logging

logger = logging.getLogger(__name__)



def run(args):
    import torch
    import torchaudio
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    print("CUDA??",torch.cuda.is_available())
    import soundfile as sf
    import datetime
    import numpy as np
    import scipy
    from tqdm import tqdm

    import utils
    import  dataset_loader, stft_loss
    import models.discriminators as discriminators
    import models.unet2d_generator as unet2d_generator
    import models.audiounet as audiounet
    import models.seanet as seanet
    from utils import do_stft
    import lowpass_utils

    path_experiment=str(args.path_experiment)

    if not os.path.exists(path_experiment):
        os.makedirs(path_experiment)
    
    #Loading data. The train dataset object is a generator. The validation dataset is loaded in memory.

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")



    ##NOT IMLEMENTED YET
    if args.bwe.generator.variant=="audiounet": #change to audiounet
        #gener_model = kuleshov_unet.Unet1d(args.unet1d).to(device)
        gener_model = audiounet.Model(mono=True).to(device)
    if args.bwe.generator.variant=="seanet": #change to seanet
        gener_model = seanet.Unet1d().to(device)
    if args.bwe.generator.variant=="unet2d":
        gener_model = unet2d_generator.Unet2d(unet_args=args.unet_generator).to(device)

    checkpoint_filepath=os.path.join(path_experiment,args.checkpoint)
       
    gener_model.load_state_dict(torch.load(checkpoint_filepath, map_location=device))
    #print("something went wrong while loading the checkpoint")

    def apply_model(x): 
        x_init=x

        if args.bwe.add_noise.add_noise:
            n=args.bwe.add_noise.power*torch.randn(x.shape)
            print("adding noise")
            x=x+n.to(device) #not tested, need to tune the noise power
       
       
        if args.bwe.generator.variant=="unet2d":
            xF =do_stft(x,win_size=args.stft.win_size, hop_size=args.stft.hop_size, device=device)
       
            with torch.no_grad():
                y_gF = gener_model(xF)
            
            y_g=utils.do_istft(y_gF, args.stft.win_size, args.stft.hop_size, device)
            y_g=y_g[:,0:x.shape[-1]]
            y_g=y_g.unsqueeze(1)
        else:
            with torch.no_grad():
                y_g = gener_model(x)

        return y_g.squeeze(1)

    audio=str(args.inference.audio)
    data, samplerate = sf.read(audio)
    #Stereo to mono
    if len(data.shape)>1:
        data=np.mean(data,axis=1)
    
    if samplerate!=22050: 
        print("Resampling")
   
        data=scipy.signal.resample(data, int((22050  / samplerate )*len(data))+1)  
 
    
    segment_size=22050*args.seg_len_s_train  #20s segment

    length_data=len(data)
    overlapsize=1024 #samples (46 ms)
    window=np.hanning(2*overlapsize)
    window_right=window[overlapsize::]
    window_left=window[0:overlapsize]
    audio_finished=False
    pointer=0
    denoised_data=np.zeros(shape=(len(data),))
    denoised_lpf=np.zeros(shape=(len(data),))
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
            if args.inference.apply_lpf:
                segment=lowpass_utils.apply_butter_lowpass_test(segment,args.inference.fc, args.fs) 
                xlpf=segment

            pred_time =apply_model(segment)
            print(pred_time.shape, segment.shape)
            residual_time=segment-pred_time

            print(segment.shape,pred_time.shape, residual_time.shape)
            segment=segment[0].detach().cpu().numpy()
            residual_time=residual_time[0].detach().cpu().numpy()
            pred_time=pred_time[0].detach().cpu().numpy()
            if args.inference.apply_lpf:
                xlpf=xlpf[0].detach().cpu().numpy()

            if pointer==0:
                pred_time=np.concatenate((pred_time[0:int(segment_size-overlapsize)], np.multiply(pred_time[int(segment_size-overlapsize):segment_size],window_right)), axis=0)
                residual_time=np.concatenate((residual_time[0:int(segment_size-overlapsize)], np.multiply(residual_time[int(segment_size-overlapsize):segment_size],window_right)), axis=0)
                if args.inference.apply_lpf:
                    xlpf=np.concatenate((xlpf[0:int(segment_size-overlapsize)], np.multiply(xlpf[int(segment_size-overlapsize):segment_size],window_right)), axis=0)
            else:
                pred_time=np.concatenate((np.multiply(pred_time[0:int(overlapsize)], window_left), pred_time[int(overlapsize):int(segment_size-overlapsize)], np.multiply(pred_time[int(segment_size-overlapsize):int(segment_size)],window_right)), axis=0)
                residual_time=np.concatenate((np.multiply(residual_time[0:int(overlapsize)], window_left), residual_time[int(overlapsize):int(segment_size-overlapsize)], np.multiply(residual_time[int(segment_size-overlapsize):int(segment_size)],window_right)), axis=0)
                if args.inference.apply_lpf:
                    xlpf=np.concatenate((np.multiply(xlpf[0:int(overlapsize)], window_left), xlpf[int(overlapsize):int(segment_size-overlapsize)], np.multiply(xlpf[int(segment_size-overlapsize):int(segment_size)],window_right)), axis=0)
                
            denoised_data[pointer:pointer+segment_size]=denoised_data[pointer:pointer+segment_size]+pred_time
            residual_noise[pointer:pointer+segment_size]=residual_noise[pointer:pointer+segment_size]+residual_time
            if args.inference.apply_lpf:
                denoised_lpf[pointer:pointer+segment_size]=denoised_lpf[pointer:pointer+segment_size]+xlpf

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
            if args.inference.apply_lpf:
                segment=lowpass_utils.apply_butter_lowpass_test(segment,args.inference.fc, args.fs) 
                xlpf=segment
            pred_time =apply_model(segment)
            residual_time=segment-pred_time
            print(segment.shape,pred_time.shape, residual_time.shape)
            segment=segment[0].detach().cpu().numpy()
            residual_time=residual_time[0].detach().cpu().numpy()
            pred_time=pred_time[0].detach().cpu().numpy()
            if args.inference.apply_lpf:
                xlpf=xlpf[0].detach().cpu().numpy()
            print(segment.shape,pred_time.shape, residual_time.shape)

            if pointer==0:
                pred_time=pred_time
                residual_time=residual_time
                if args.inference.apply_lpf:
                    xlpf=xlpf
            else:
                pred_time=np.concatenate((np.multiply(pred_time[0:int(overlapsize)], window_left), pred_time[int(overlapsize):int(segment_size)]),axis=0)
                residual_time=np.concatenate((np.multiply(residual_time[0:int(overlapsize)], window_left), residual_time[int(overlapsize):int(segment_size)]),axis=0)
                if args.inference.apply_lpf:
                    xlpf=np.concatenate((np.multiply(xlpf[0:int(overlapsize)], window_left), xlpf[int(overlapsize):int(segment_size)]),axis=0)

            denoised_data[pointer::]=denoised_data[pointer::]+pred_time[0:lensegment]
            residual_noise[pointer::]=residual_noise[pointer::]+residual_time[0:lensegment]
            if args.inference.apply_lpf:
                denoised_lpf[pointer::]=denoised_lpf[pointer::]+xlpf[0:lensegment]

    basename=os.path.splitext(audio)[0]
    wav_noisy_name=basename+"_"+args.inference.exp_name+"_bwe_input"+".wav"
    sf.write(wav_noisy_name, data, 22050)
    wav_output_name=basename+"_"+args.inference.exp_name+"_bwe_output"+".wav"
    sf.write(wav_output_name, denoised_data, 22050)
    #wav_output_name=basename+"_"+args.inference.exp_name+"_bwe_residual"+".wav"
    #sf.write(wav_output_name, residual_noise, 22050)
    
    if args.inference.apply_lpf:
        wav_output_name=basename+"_"+args.inference.exp_name+"_bwe_lpf"+".wav"
       
        sf.write(wav_output_name, denoised_lpf, 22050)


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







