import torch 
import numpy as np
import cv2
import torchaudio
from collections import OrderedDict

#def load_denoiser_model(device,args):
#
#    denoiser_model = unet.MultiStage_denoise(unet_args=args.unet)
#    denoiser_model.to(device)
#    checkpoint_denoiser=args.denoiser.checkpoint
#    try:
#        denoiser_model.load_state_dict(torch.load(checkpoint_denoiser, map_location=device))
#    except:
#        print("Force loading!!!")
#        check_point = torch.load(checkpoint_denoiser, map_location=device)
#        check_point.key()
#        new_state_dict = OrderedDict()
#        for k, v in state_dict.items():
#            name = k[7:] # remove 'module.' of dataparallel
#            new_state_dict[name]=v
#        denoiser_model.load_state_dict(new_state_dict)
#
#    return denoiser_model
#
def add_high_freqs(data, f_dim=1025):
    dims=list(data.size())
    dims[3]=1025
    b=torch.zeros(dims) 
    b=b.to(data.device)
    data=torch.cat((data[:,:,:,0:f_dim],b[:,:,:,0:1025-f_dim]),3)
    return data

def do_istft(data, win_size=2048, hop_size=512, device="cuda"):
    window=torch.hamming_window(window_length=win_size)
    window=window.to(device)
    data=data.permute(0,3,2,1)
    pred_time=torch.istft(data, win_size, hop_length=hop_size,  window=window, center=False, return_complex=False)
    return pred_time



def do_stft(noisy, win_size=2048, hop_size=512, device="cpu"):
    
    window=torch.hamming_window(window_length=win_size)
    window=window.to(noisy.device)
    noisy=torch.cat((noisy, torch.zeros(noisy.shape[0],win_size).to(noisy.device)),1)
    stft_signal_noisy=torch.stft(noisy, win_size, hop_length=hop_size,window=window,center=False,return_complex=False)
    stft_signal_noisy=stft_signal_noisy.permute(0,3,2,1)
    
    return stft_signal_noisy

def computeSNR(ref,x): #check shapes
    # Signal-to-noise ratio for torch tensors
    ref_pow = (ref**2).mean(-1) + np.finfo('float32').eps
    dif_pow = ((x - ref)**2).mean(-1) + np.finfo('float32').eps
    snr_val = 10 * torch.log10(ref_pow / dif_pow)
    snr_val= torch.sum(snr_val,axis=0)
    return snr_val

def computeLSD(y,y_g): #check!!!
    yF=do_stft(y,win_size=2048, hop_size=512)
    ygF=do_stft(y_g, win_size=2048, hop_size=512)
    yF=torch.sqrt(yF[:,0,:,:]**2 +yF[:,1,:,:]**2)
    ygF=torch.sqrt(ygF[:,0,:,:]**2 +ygF[:,1,:,:]**2)
    Sy = torch.log10(torch.abs(yF)**2 + 1e-8)
    Syg = torch.log10(torch.abs(ygF)**2 + 1e-8)
    lsd = torch.sum(torch.mean(torch.sqrt(torch.mean((Sy-Syg)**2 + 1e-8, axis=2)), axis=1), axis=0)

    return lsd

  
def generate_images(audio):
    audio=audio.cpu()
    audio=audio.unsqueeze(0)
    cpx=do_stft(audio, win_size=1025, hop_size=256)
    cpx=cpx.permute(0,3,2,1)
    cpx=cpx.cpu().detach().numpy()
    spectro=np.clip((np.flipud(np.transpose(10*np.log10(np.sqrt(np.power(cpx[...,0],2)+np.power(cpx[...,1],2)))))+30)/50,0,1)
    cmap=cv2.COLORMAP_JET
    spectro = np.array((1-spectro)* 255, dtype = np.uint8)
    spectro = cv2.applyColorMap(spectro, cmap)
    return np.flipud(np.fliplr(spectro))

