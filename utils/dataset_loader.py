import ast

#import tensorflow as tf
import random
import scipy.signal
import os
import numpy as np
import soundfile as sf
import math
import pandas as pd
import glob
from tqdm import tqdm
import torch

#generator function. It reads the csv file with pandas and loads the largest audio segments from each recording. If extend=False, it will only read the segments with length>length_seg, trim them and yield them with no further processing. Otherwise, if the segment length is inferior, it will extend the length using concatenative synthesis.


def generate_real_recordings_data(path_recordings, fs=44100, seg_len_s=15, stereo=False):

    records_info=os.path.join(path_recordings,"audio_files.txt")
    num_lines = sum(1 for line in open(records_info))
    f = open(records_info,"r")
    #load data record files
    print("Loading record files")
    records=[]
    seg_len=fs*seg_len_s
    pointer=int(fs*5) #starting at second 5 by default
    for i in tqdm(range(num_lines)):
        audio=f.readline() 
        audio=audio[:-1]
        data, fs=sf.read(os.path.join(path_recordings,audio))
        if len(data.shape)>1 and not(stereo):
            data=np.mean(data,axis=1)
        #elif stereo and len(data.shape)==1:
        #    data=np.stack((data, data), axis=1)

        #normalize
        data=data/np.max(np.abs(data))
        segment=data[pointer:pointer+seg_len]
        records.append(segment.astype("float32"))

    return records

def generate_paired_data_test_formal(path_pianos, path_noises, noise_amount="low_snr",num_samples=-1, fs=44100, seg_len_s=5 , extend=True, stereo=False, prenoise=False):

    print(num_samples)
    segments_clean=[]
    segments_noisy=[]
    seg_len=fs*seg_len_s
    noises_info=os.path.join(path_noises,"info.csv")
    np.random.seed(42)
    if noise_amount=="low_snr":
        SNRs=np.random.uniform(2,6,num_samples)
    elif noise_amount=="mid_snr":
        SNRs=np.random.uniform(6,12,num_samples)

    scales=np.random.uniform(-4,0,num_samples)
    #SNRs=[2,6,12] #HARDCODED!!!!
    i=0
    print(path_pianos[0])
    print(seg_len)
    train_samples=glob.glob(os.path.join(path_pianos[0],"*.wav"))
    train_samples=sorted(train_samples)

    if prenoise:
        noise_generator=noise_sample_generator(noises_info,fs, seg_len+fs, extend, "test") #Adds 1s of silence add the begiing, longer noise
    else:
        noise_generator=noise_sample_generator(noises_info,fs, seg_len, extend, "test") #this will take care of everything
    #load data clean files
    for file in tqdm(train_samples):  #add [1:5] for testing
        data_clean, samplerate = sf.read(file)
        if samplerate!=fs: 
            print("!!!!WRONG SAMPLE RATe!!!")
        #Stereo to mono
        if len(data_clean.shape)>1 and not(stereo):
            data_clean=np.mean(data_clean,axis=1)
        #elif stereo and len(data_clean.shape)==1:
        #   data_clean=np.stack((data_clean, data_clean), axis=1)
        #normalize
        data_clean=data_clean/np.max(np.abs(data_clean))
        #data_clean_loaded.append(data_clean)
 
        #framify data clean files
 
        #framify  arguments: seg_len, hop_size
        hop_size=int(seg_len)# no overlap
 
        num_frames=np.floor(len(data_clean)/hop_size - seg_len/hop_size +1) 
        print(num_frames)
        if num_frames==0:
            data_clean=np.concatenate((data_clean, np.zeros(shape=(int(2*seg_len-len(data_clean)),))), axis=0)
            num_frames=1

        data_not_finished=True
        pointer=0
        while(data_not_finished):
            if i>=num_samples:
                break
            segment=data_clean[pointer:pointer+seg_len]
            pointer=pointer+hop_size
            if pointer+seg_len>len(data_clean):
                data_not_finished=False
            segment=segment.astype('float32')
    
            #SNRs=np.random.uniform(2,20)
            snr=SNRs[i] 
            scale=scales[i]
            #load noise signal
            data_noise= next(noise_generator)
            data_noise=np.mean(data_noise,axis=1)
            #normalize
            data_noise=data_noise/np.max(np.abs(data_noise))
            new_noise=data_noise #if more processing needed, add here
            #load clean data
            #configure sizes
            power_clean=np.var(segment)
            #estimate noise power
            if prenoise:
                power_noise=np.var(new_noise[fs::])
            else:
                power_noise=np.var(new_noise)

            snr = 10.0**(snr/10.0)

            #sum both signals according to snr
            if prenoise:
                segment=np.concatenate((np.zeros(shape=(fs,)),segment),axis=0) #add one second of silence
            noise_signal=np.sqrt(power_clean/(snr*power_noise))*new_noise #not sure if this is correct, maybe revisit later!!

            noise_signal=noise_signal.astype('float32')
            #yield tf.convert_to_tensor(summed), tf.convert_to_tensor(segment)
  
                
            noise_signal=10.0**(scale/10.0) *noise_signal
            segment=10.0**(scale/10.0) *segment
            segments_noisy.append(noise_signal.astype('float32'))
            segments_clean.append(segment.astype('float32'))
            i=i+1

    return segments_noisy, segments_clean

def generate_test_data(path_music, path_noises,num_samples=-1, fs=44100, seg_len_s=5):

    segments_clean=[]
    segments_noisy=[]
    seg_len=fs*seg_len_s
    noises_info=os.path.join(path_noises,"info.csv")
    SNRs=[2,6,12] #HARDCODED!!!!
    for path in path_music:
        print(path)
        train_samples=glob.glob(os.path.join(path,"*.wav"))
        train_samples=sorted(train_samples)

        noise_generator=noise_sample_generator(noises_info,fs, seg_len, "test") #this will take care of everything
        #load data clean files
        jj=0
        for file in tqdm(train_samples):  #add [1:5] for testing
            data_clean, samplerate = sf.read(file)
            if samplerate!=fs: 
                print("!!!!WRONG SAMPLE RATe!!!")
            #Stereo to mono
            if len(data_clean.shape)>1:
                data_clean=np.mean(data_clean,axis=1)
            #normalize
            data_clean=data_clean/np.max(np.abs(data_clean))
            #data_clean_loaded.append(data_clean)
     
            #framify data clean files
     
            #framify  arguments: seg_len, hop_size
            hop_size=int(seg_len)# no overlap
     
            num_frames=np.floor(len(data_clean)/hop_size - seg_len/hop_size +1) 
            if num_frames==0:
                data_clean=np.concatenate((data_clean, np.zeros(shape=(int(2*seg_len-len(data_clean)),))), axis=0)
                num_frames=1

            pointer=0
            segment=data_clean[pointer:pointer+(seg_len-2*fs)]
            segment=segment.astype('float32')
            segment=np.concatenate(( np.zeros(shape=(2*fs,)), segment), axis=0) #I hope its ok
            #segments_clean.append(segment)
        
            for snr in SNRs:
                #load noise signal
                data_noise= next(noise_generator)
                data_noise=np.mean(data_noise,axis=1)
                #normalize
                data_noise=data_noise/np.max(np.abs(data_noise))
                new_noise=data_noise #if more processing needed, add here
                #load clean data
                #configure sizes
                #estimate clean signal power
                power_clean=np.var(segment)
                #estimate noise power
                power_noise=np.var(new_noise)

                snr = 10.0**(snr/10.0)

                #sum both signals according to snr
                summed=segment+np.sqrt(power_clean/(snr*power_noise))*new_noise #not sure if this is correct, maybe revisit later!!
                summed=summed.astype('float32')
                #yield tf.convert_to_tensor(summed), tf.convert_to_tensor(segment)
      
                segments_noisy.append(summed.astype('float32'))
                segments_clean.append(segment.astype('float32'))

    return segments_noisy, segments_clean

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, path_music,  fs=44100, seg_len_s=5):
                
        test_samples=glob.glob(os.path.join(path_music,"*.wav"))

        self.records=[]
        seg_len=int(fs*seg_len_s)
        pointer=int(fs*5) #starting at second 5 by default
        for i in tqdm(range(len(test_samples))):
            data, sr=sf.read(test_samples[i])
            if len(data.shape)>1 and not(stereo):
                data=np.mean(data,axis=1)
            if sr !=fs: 
                
                print("resampling", sr, fs, test_samples[i])
                data=scipy.signal.resample(data, int(len(data)*fs/sr))

            #normalize
            data=0.9*(data/np.max(np.abs(data)))

            segment=data[pointer:pointer+seg_len]

            self.records.append(segment.astype("float32"))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]

#MAYBE eliminate
class TestRealDataset(torch.utils.data.Dataset):
    def __init__(self, path_music, fs=44100, seg_len_s=5):
        val_samples=[]
        for path in path_music:
            val_samples.extend(glob.glob(os.path.join(path,"*.wav")))
    
        #load data clean files
        print("Loading clean files")
        data_clean_loaded=[]
        for ff in tqdm(range(0,len(val_samples))):  #add [1:5] for testing
            data_clean, samplerate = sf.read(val_samples[ff])
            if samplerate!=fs: 
                print("!!!!WRONG SAMPLE RATe!!!")
            #Stereo to mono
            if len(data_clean.shape)>1 :
                data_clean=np.mean(data_clean,axis=1)
            #normalize
            data_clean=data_clean/np.max(np.abs(data_clean))
            data_clean_loaded.append(data_clean)
            del data_clean
    
        #framify data clean files
        print("Framifying clean files")
        seg_len=fs*seg_len_s
        self.segments_clean=[]
        for file in tqdm(data_clean_loaded):
    
            #framify  arguments: seg_len, hop_size
            hop_size=int(seg_len)# no overlap
    
            num_frames=np.floor(len(file)/hop_size - seg_len/hop_size +1) 
            pointer=0
            for i in range(0,int(num_frames)):
                segment=file[pointer:pointer+seg_len]
                pointer=pointer+hop_size
                segment=segment.astype('float32')
                self.segments_clean.append(segment)
    
        del data_clean_loaded
        
        SNRs=np.random.uniform(2,20,len(self.segments_clean))
        scales=np.random.uniform(-6,4,len(self.segments_clean))
        #noise_shapes=np.random.randint(0,len(noise_samples), len(segments_clean))
        noises_info=os.path.join(path_noises,"info.csv")
    
        noise_generator=noise_sample_generator(noises_info,fs, seg_len,  "validation") #this will take care of everything
        
    
        #generate noisy segments
        #load noise samples using pandas dataframe. Each split (train, val, test) should have its unique csv info file
    
        #noise_samples=glob.glob(os.path.join(path_noises,"*.wav"))
        self.segments_noisy=[]
        print("Processing noisy segments")
    
        for i in tqdm(range(0,len(self.segments_clean))):
            #load noise signal
            data_noise= next(noise_generator)
            #Stereo to mono
            data_noise=np.mean(data_noise,axis=1)
            #normalize
            data_noise=data_noise/np.max(np.abs(data_noise))
            new_noise=data_noise #if more processing needed, add here
            #load clean data
            data_clean=self.segments_clean[i]
            #configure sizes
            
             
            #estimate clean signal power
            power_clean=np.var(data_clean)
            #estimate noise power
            power_noise=np.var(new_noise)
    
            snr = 10.0**(SNRs[i]/10.0)
    
            #sum both signals according to snr
            noise_signal=np.sqrt(power_clean/(snr*power_noise))*new_noise #not sure if this is correct, maybe revisit later!!
                #the rest is normal
            
            noise_signal=10.0**(scales[i]/10.0) *noise_signal
            self.segments_clean[i]=10.0**(scales[i]/10.0) *self.segments_clean[i]
    
            self.segments_noisy.append(noise_signal.astype('float32'))
            

    def __len__(self):
        return len(self.segments_clean)

    def __getitem__(self, idx):
        return self.segments_noisy[idx], self.segments_clean[idx]

class ValDataset(torch.utils.data.Dataset):

    def __init__(self, path_music, path_noises, fs=44100, seg_len_s=5):
        val_samples=[]
        for path in path_music:
            val_samples.extend(glob.glob(os.path.join(path,"*.wav")))
    
        #load data clean files
        print("Loading clean files")
        data_clean_loaded=[]
        for ff in tqdm(range(0,len(val_samples))):  
            data_clean, samplerate = sf.read(val_samples[ff])
            if samplerate!=fs: 
                print("!!!!WRONG SAMPLE RATe!!!")
            #Stereo to mono
            if len(data_clean.shape)>1 :
                data_clean=np.mean(data_clean,axis=1)
            #normalize
            data_clean=data_clean/np.max(np.abs(data_clean))
            data_clean_loaded.append(data_clean)
            del data_clean
    
        #framify data clean files
        print("Framifying clean files")
        seg_len=int(fs*seg_len_s)
        self.segments_clean=[]
        for file in tqdm(data_clean_loaded):
    
            #framify  arguments: seg_len, hop_size
            hop_size=int(seg_len)# no overlap
    
            num_frames=np.floor(len(file)/hop_size - seg_len/hop_size +1) 
            pointer=0
            for i in range(0,int(num_frames)):
                segment=file[pointer:pointer+int(seg_len)]
                pointer=pointer+hop_size
                segment=segment.astype('float32')
                self.segments_clean.append(segment)
    
        del data_clean_loaded
        
        scales=np.random.uniform(-6,4,len(self.segments_clean))

        self.segments_clean[i]=10.0**(scales[i]/10.0) *self.segments_clean[i]

    def __len__(self):
        return len(self.segments_clean)

    def __getitem__(self, idx):
        return self.segments_clean[idx]

        
#Train dataset object

class TrainDataset (torch.utils.data.IterableDataset):
    def __init__(self, path_music,  path_noises,  fs=44100, seg_len_s=5,seed=42 ):
        super(TrainDataset).__init__()
        random.seed(seed)
        np.random.seed(seed)
        self.train_samples=[]
        for path in path_music:
            self.train_samples.extend(glob.glob(os.path.join(path ,"*.wav")))
       
        self.seg_len=int(fs*seg_len_s)
        self.fs=fs

    def __iter__(self):
        while True:
            random.shuffle(self.train_samples)
            for file in self.train_samples:  
                data, samplerate = sf.read(file)
                assert(samplerate==self.fs, "wrong sampling rate")
                data_clean=data
                #Stereo to mono
                if len(data.shape)>1 :
                    data_clean=np.mean(data_clean,axis=1)
    
                #normalize
                data_clean=data_clean/np.max(np.abs(data_clean))
         
                #framify data clean files
         
                #framify  arguments: seg_len, hop_size
                hop_size=int(self.seg_len)
         
                num_frames=np.floor(len(data_clean)/self.seg_len) 
                if num_frames==0:
                    data_clean=np.concatenate((data_clean, np.zeros(shape=(int(2*self.seg_len-len(data_clean)),))), axis=0)
                    num_frames=1
                    pointer=0
                    data_clean=np.roll(data_clean, np.random.randint(0,self.seg_len)) #if only one frame, roll it for augmentation
                elif num_frames>1:
                    pointer=np.random.randint(0,hop_size)  #initial shifting, graeat for augmentation, better than overlap as we get different frames at each "while" iteration
                else:
                    pointer=0
    
                data_not_finished=True

                while(data_not_finished):

                    segment=data_clean[pointer:pointer+self.seg_len]
                    pointer=pointer+hop_size

                    if pointer+self.seg_len>len(data_clean):
                        data_not_finished=False

                    segment=segment.astype('float32')
            
                    scale=np.random.uniform(-6,4)
            
                    segment=10.0**(scale/10.0) *segment
                        
                    yield  segment



def load_real_test_recordings(buffer_size, path_recordings,   **kwargs):
    print("Generating real test dataset")
        
    segments_noisy=generate_real_recordings_data(path_recordings, **kwargs)

    dataset_test=tf.data.Dataset.from_tensor_slices(segments_noisy)
    #train_dataset = train.cache().shuffle(buffer_size).take(info.splits["train"].num_examples)
    return dataset_test
