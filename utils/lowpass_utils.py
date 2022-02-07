import scipy.signal 
import numpy as np
import torch
import torchaudio
import math

import pandas


def get_butter_filters(filters_csv="butter_filters.csv"):
    filter_data = np.genfromtxt(filters_csv, delimiter=',')
    print(filter_data.dtype)
    print(filter_data.shape)
    return filter_data


def get_random_FIR_filters(num_filters,mean_fc=3500, std_fc= 300,args=None, sr=44100, seed=43, device="cuda"):
    np.random.seed(seed)
    
    filters=[]
    order=25
    fc=np.random.normal(mean_fc, std_fc, num_filters)

    for i in range(num_filters):
        B =get_FIR_lowpass(order,fc[i],1, sr)
        filters.append( B)
    return filters

def get_FIR_lowpass(order,fc, beta, sr):
    """ Gets coeffcients of chosen filter
    
    Arguments:
        cutoff {int} -- cutoff frequency
    
    Keyword Arguments:
        sr {int} -- sampling rate of the input signal (default: {44100})
    
    Returns:
        numpy 1d array -- lowpassed signal
    """
    B=scipy.signal.firwin(numtaps=order,cutoff=fc, width=beta,window="kaiser", fs=sr)
    return B
        
def get_FIR_lowpass_test(order,fc, beta, sr):
    """ Gets coeffcients of chosen filter
    
    Arguments:
        cutoff {int} -- cutoff frequency
    
    Keyword Arguments:
        sr {int} -- sampling rate of the input signal (default: {44100})
        filter_type {str} -- type of filter, only butter and cheby1 are implemented (default: {'butter'})
    
    Returns:
        numpy 1d array -- lowpassed signal
    """
    B=scipy.signal.firwin(numtaps=order,cutoff=fc, width=beta,window="kaiser", fs=sr)
    return B

def apply_butter_lowpass_test(x,fc, sr):
    #B= get_FIR_lowpass_test(25,11000, 1, sr)
    #B=torch.from_numpy(B)
    #B=B.to(x.dtype)
    #B=B.to(x.device)
    #B=B.unsqueeze(0)
    #B=B.unsqueeze(0)
    #B=B.unsqueeze(0)
    #print(B.shape)
    #rint(x.shape)
    #x=x.unsqueeze(1)
    
    #x=torch.nn.functional.conv1d(x,B,padding="same", groups=x.shape[1])
    #x=x.squeeze(1)
        
    b, a = scipy.signal.butter(6, fc, 'low', fs=sr)
    #xx=x.cpu().numpy()
    #y=scipy.signal.filtfilt(b,a,xx) 
    ##print(y.shape, xx.shape)
    #y=torch.from_numpy(y.copy())
    #y=y.to(x.dtype)
    #y=y.to(x.device)
    
    a_s= torch.from_numpy(a)
    a_s=a_s.to(x.dtype)
    a_s=a_s.to(x.device)
    b_s= torch.from_numpy(b)
    b_s=b_s.to(x.dtype)
    b_s=b_s.to(x.device)
    y=torchaudio.functional.lfilter(x, a_coeffs=a_s, b_coeffs= b_s) 
    return y



def gauss_f(f_x,F,Noct):
    
    #GAUSS_F calculate frequency-domain Gaussian with unity gain

    #G = GAUSS_F(F_X,F,NOCT) calculates a frequency-domain Gaussian function
    #for frequencies F_X, with centre frequency F and bandwidth F/NOCT.

    sigma = ((F+1e-10)/Noct)/math.pi; # standard deviation
    g = torch.exp(-(torch.pow(f_x-F,2)/(2*(torch.pow(sigma,2))))); #Gaussian
    g = g/torch.sum(g); # normalise magnitude

    return g

def equalize_sample(sample,world_curve,NFFT=4096, hop=2048, cutoff=3500, Fs=44100, lp_filter=None):


    if NFFT %2==0:
        Nout = (NFFT/2)+1;
    else:
        Nout = (NFFT+1)/2;

    f=torch.linspace(0,int(Nout)-1,int(Nout))*Fs/NFFT
    f=f.to(sample.device)
    fi=torch.argmin(torch.abs(f-cutoff))
    
    win=torch.hamming_window(window_length=NFFT)
    win=win.to(sample.device)
    samplec=torch.cat((sample, torch.zeros(sample.shape[0],NFFT).to(sample.device)),1)
    oldS=torch.stft(samplec,n_fft=NFFT,hop_length=2048,window=win, center=False, return_complex=True, onesided=False)

    oldSmean=torch.abs(oldS[:,0:2049,0:-1])*(torch.sum(win)/(NFFT**2))
    oldSmean=torch.mean(oldSmean,2)
    oldSmean=20*torch.log10(oldSmean)
    
    Noct=1
    z_oct=torch.zeros_like(oldSmean)

    for i in range(0,len(f)): #too slow!!!
        g=gauss_f(f,f[i],Noct)
        z_oct[:,i]=torch.sum(torch.mul(g,oldSmean), 1)

    diffs=world_curve-z_oct
    for j in range(diffs.shape[0]):
        diffs[j,fi::]=diffs[j,fi]
    #diffs=torch.clamp(diffs,-30,30) #patch to avoid NaNs
    diffs=10**(diffs/20)
    a=torch.conj(torch.flip(diffs[:,1:-1],(1,)))
    H=torch.cat((diffs,a),1)
    oldS_filtered=torch.mul(oldS.permute(2,0,1),H)
    oldS_filtered=oldS_filtered.permute(1,2,0)

    equalized_sample=torch.istft(oldS_filtered,n_fft=NFFT,hop_length=2048, window=win, center=False, onesided=False)
    equalized_sample=equalized_sample[:,0:sample.shape[-1]]

    if lp_filter==None:
        return equalized_sample.type(sample.dtype), H
    else:
        oldS_lpf=torch.mul(oldS_filtered.permute(2,0,1),lp_filter)
        oldS_lpf=oldS_lpf.permute(1,2,0)
        equalized_and_filtered=torch.istft(oldS_lpf,n_fft=NFFT,hop_length=2048, window=win, center=False, onesided=False)
        equalized_and_filtered=equalized_and_filtered[:,0:sample.shape[-1]]
        return equalized_sample.type(sample.dtype), equalized_and_filtered.type(sample.dtype), H

def use_only_butterworth(sample, lp_filter, NFFT=4096):

    win=torch.hamming_window(window_length=NFFT)
    win=win.to(sample.device)
    samplec=torch.cat((sample, torch.zeros(sample.shape[0],NFFT).to(sample.device)),1)
    oldS=torch.stft(samplec,n_fft=NFFT,hop_length=2048,window=win, center=False, return_complex=True, onesided=False)

    oldS_lpf=torch.mul(oldS.permute(2,0,1),lp_filter)
    oldS_lpf=oldS_lpf.permute(1,2,0)
    equalized_and_filtered=torch.istft(oldS_lpf,n_fft=NFFT,hop_length=2048, window=win, center=False, onesided=False)
    equalized_and_filtered=equalized_and_filtered[:,0:sample.shape[-1]]
    return equalized_and_filtered.type(sample.dtype)

def invert_equalization(sample, H,NFFT=4096, hop=2048):
    win=torch.hamming_window(window_length=NFFT)
    win=win.to(sample.device)
    samplec=torch.cat((sample, torch.zeros(sample.shape[0],NFFT).to(sample.device)),1)
    oldS_recover=torch.stft(samplec,n_fft=NFFT,hop_length=2048,window=win, center=False, return_complex=True, onesided=False)
    oldS_rec_filter=torch.mul(oldS_recover.permute(2,0,1),(1/H))
    oldS_rec_filter=oldS_rec_filter.permute(1,2,0)
    oldS_recovered_time=torch.istft(oldS_rec_filter,n_fft=NFFT,hop_length=2048, window=win, center=False, onesided=False)
    oldS_recovered_time=oldS_recovered_time[:,0:sample.shape[-1]]
    return oldS_recovered_time

def load_world_curve():
    world_curve=np.genfromtxt('/scratch/work/molinee2/unet_dir/bwe_pytorch/world_curve.csv', delimiter=',')
    world_curve=torch.from_numpy(world_curve)
    return world_curve

def get_butterworth_lowpass(cutoff,N=4096, Fs=44100):

    b,a =scipy.signal.butter(20,cutoff, fs=Fs)
    delta=np.zeros((N,)); 
    delta[0]=1;
    firvut=scipy.signal.lfilter(b,a,delta);
    firvut=torch.from_numpy(firvut)
    FF=torch.fft.fft(firvut)
    return FF

def get_ideal_filter(cutoff,N=4096, Fs=44100):
    F=torch.ones((int(N/2 +1),))
    F[int(cutoff*N/Fs)::]=0
    a=torch.conj(torch.flip(F[1:-1],(0,)))
    FF =torch.cat((F,a),0)
    return FF
def get_lp_filters( filters, bs):
    B=[]
    ii=np.random.randint(len(filters), size=bs)
    for j in range(bs):
        k=ii[j]
        B.append(filters[k])
    B=torch.FloatTensor(B)
    return B
def apply_lpf(y,B):
    B=B.unsqueeze(1)
    y=y.unsqueeze(0)
    #weight=torch.nn.Parameter(B)
    y_lpf=torch.nn.functional.conv1d(y,B,padding="same", groups=y.shape[1])
    y=y.squeeze(0)
    y_lpf=y_lpf.squeeze(0) #some redundancy here, but its ok
    return y_lpf

def apply_low_pass(y,filters,args):

    ii=np.random.randint(args.bwe.lpf.num_random_filters)
    #ii=1
    B=filters[ii]
    B=torch.FloatTensor(B).to(y.device)
    B=B.unsqueeze(0)
    B=B.unsqueeze(0)
    y=y.unsqueeze(1)
    #weight=torch.nn.Parameter(B)
    y_lpf=torch.nn.functional.conv1d(y,B,padding="same")
    y=y.squeeze(1)
    y_lpf=y_lpf.squeeze(1) #some redundancy here, but its ok
    #y_lpf=y

    return y_lpf
