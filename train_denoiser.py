import os
import hydra
import logging

logger = logging.getLogger(__name__)



def run(args):
    import models.denoiser as denoiser
    #import tensorflow as tf
    import torch
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    print("CUDA??",torch.cuda.is_available())
    import  utils.dataset_loader as dataset_loader
    #from tensorflow.keras.optimizers import Adam
    import soundfile as sf
    import datetime
    import random
    from tqdm import tqdm
    import numpy as np

    dirname = os.path.dirname(__file__)
    path_experiment = os.path.join(dirname, str(args.path_experiment))

    if not os.path.exists(path_experiment):
        os.makedirs(path_experiment)
    
    path_music_train=args.dset.path_music_train
    #path_music_test=args.dset.path_music_test
    #path_music_validation=args.dset.path_music_validation

    path_noise=args.dset.path_noise
    #path_recordings=args.dset.path_recordings
    
    fs=args.fs
    overlap=args.overlap
    seg_len_s_train=args.seg_len_s_train

    batch_size=args.batch_size
    epochs=args.epochs

    num_real_test_segments=args.num_real_test_segments
    buffer_size=args.buffer_size #for shuffle
    
    tensorboard_logs=args.tensorboard_logs
    



    def do_stft(noisy, clean=None):
        
        #window_fn = tf.signal.hamming_window
        win_size=args.stft.win_size
        hop_size=args.stft.hop_size
        window=torch.hamming_window(window_length=win_size)
        window=window.to(device)

        noisy=torch.cat((noisy, torch.zeros(args.batch_size,win_size).to(device)), 1)
        stft_signal_noisy=torch.stft(noisy, win_size, hop_length=hop_size,window=window,center=False,return_complex=False)
        stft_signal_noisy=stft_signal_noisy.permute(0,3,2,1)

        
        if clean!=None:

            clean=torch.cat((clean, torch.zeros(args.batch_size,win_size).to(device)), 1)
            stft_signal_clean=torch.stft(clean, win_size, hop_length=hop_size,window=window, center=False,return_complex=False)
            stft_signal_clean=stft_signal_clean.permute(0,3,2,1)

            return stft_signal_noisy, stft_signal_clean
        else:
    
            return stft_signal_noisy
    
    #Loading data. The train dataset object is a generator. The validation dataset is loaded in memory.
    dataset_train=dataset_loader.TrainDataset( path_music_train, path_noise, fs,seg_len_s_train, seed=0)

    #dataset_val=dataset_loader.ValDataset(path_music_validation, path_noise, fs,seg_len_s_train)
    #dataset_test=dataset_loader.TestDataset(path_music_test, path_noise, fs,seg_len_s_train)
    #train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    def worker_init_fn(worker_id):                                                          
        np.random.seed(np.random.get_state()[1][0] + worker_id)
        random.seed(np.random.get_state()[1][0] + worker_id) #not tested

    train_loader=DataLoader(dataset_train,num_workers=args.num_workers, batch_size=args.batch_size, worker_init_fn=worker_init_fn)
    
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet_model = denoiser.MultiStage_denoise(unet_args=args.denoiser)

    unet_model.to(device)


    if args.use_tensorboard:
        log_dir = os.path.join(tensorboard_logs, os.path.basename(path_experiment)+"_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"_"+str(0))
        train_summary_writer = SummaryWriter(log_dir+"/train")
        val_summary_writer = SummaryWriter(log_dir+"/validation")
    
    #path where the checkpoints will be saved
    checkpoint_filepath=os.path.join(path_experiment, 'checkpoint')
    

    iterator = iter(train_loader)

    loss = torch.nn.L1Loss()
    current_lr=args.lr
    optimizer = torch.optim.Adam(unet_model.parameters(),lr=current_lr, betas=(args.beta1,args.beta2))
    optimizer.zero_grad()

    for epoch in range(epochs):
        train_loss=0
        step_loss=0
        #train_sampler.set_epoch(epoch)
        for step in tqdm(range(int(args.steps_per_epoch)), desc="Training epoch "+str(epoch)):
            
            noisy, clean=iterator.next()    
            noisy=noisy.to(device)
            clean=clean.to(device)

            noisyF, cleanF=do_stft(noisy, clean)

            if args.unet.num_stages==1:     
                y_predF_s1=unet_model(noisyF)

                loss_s1=loss(y_predF_s1,cleanF)
        
                loss_total=loss_s1.mean()
            elif args.unet.num_stages>1:
                y_predF_s2,y_predF_s1=unet_model(noisyF)
                
                loss_s1=loss(y_predF_s1,cleanF)
                loss_s2=loss(y_predF_s2,cleanF)
        
                loss_total=loss_s1.mean()+loss_s2.mean()

            loss_total.backward()
            if (step+1)%args.multi_batch == 0:
                optimizer.step()
                optimizer.zero_grad()
            step_loss=loss_total.item()
            train_loss += loss_total.item()


            train_summary_writer.add_scalar('batch_loss', step_loss, int(step+epoch*(args.steps_per_epoch)))



        template = ("Epoch {}, Loss: {}")
        print (template.format(epoch+1, train_loss))

        train_summary_writer.add_scalar('epoch_loss', train_loss, epoch)

         
        if (epoch+1) % 12 == 0: #modify this
            if args.variable_lr:
                current_lr*=1e-1
                for g in optimizer.param_groups:
                    g['lr'] = current_lr

        if (epoch+1) % args.freq_inference == 0:
            print(checkpoint_filepath)
            torch.save(unet_model.state_dict(), checkpoint_filepath+"_"+str(epoch))

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







