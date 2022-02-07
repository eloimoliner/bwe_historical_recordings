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
    import datetime
    from tqdm import tqdm
    import numpy as np

    import utils
    #import hifi-gan scripts
    import models.discriminators as discriminators
    import models.unet2d_generator as unet2d_generator
    import models.audiounet as audiounet
    import models.seanet as seanet
    import utils.utils as utils 
    import utils.lowpass_utils as lowpass_utils 
    import  utils.dataset_loader as dataset_loader
    import  utils.stft_loss as stft_loss

    path_experiment=str(args.path_experiment)

    if not os.path.exists(path_experiment):
        os.makedirs(path_experiment)
    

    #Loading data. The train dataset object is a generator. The validation and test datasets are loaded in memory.
    dataset_train=dataset_loader.TrainDataset( args.dset.path_music_train, args.dset.path_noise, args.fs,args.seg_len_s_train)

    dataset_val=dataset_loader.ValDataset(args.dset.path_music_validation, args.dset.path_noise, args.fs,args.seg_len_s_val)
    dataset_test=dataset_loader.TestDataset(args.dset.path_recordings, args.fs,args.seg_len_s_test)

    def worker_init_fn(worker_id):                                                          
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    train_loader=DataLoader(dataset_train,num_workers=args.num_workers, batch_size=args.batch_size,  worker_init_fn=worker_init_fn)
    iterator = iter(train_loader)
    
    val_loader=DataLoader(dataset_val,num_workers=args.num_workers, batch_size=args.batch_size_val,  worker_init_fn=worker_init_fn)
    test_loader=DataLoader(dataset_test,num_workers=args.num_workers, batch_size=1)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


    if args.bwe.generator.variant=="audiounet": #change to audiounet
        #gener_model = kuleshov_unet.Unet1d(args.unet1d).to(device)
        gener_model = audiounet.Model(mono=True).to(device)
    if args.bwe.generator.variant=="seanet": #change to seanet
        gener_model = seanet.Unet1d().to(device)
    if args.bwe.generator.variant=="unet2d":
        gener_model = unet2d_generator.Unet2d(unet_args=args.unet_generator).to(device)


    msd = discriminators.MultiScaleDiscriminator().to(device)

    stft_loss = stft_loss.MultiResolutionSTFTLoss()

    if args.use_tensorboard:
        log_dir = os.path.join(args.tensorboard_logs, os.path.basename(path_experiment)+"_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        train_summary_writer = SummaryWriter(log_dir+"/train")
        val_summary_writer = SummaryWriter(log_dir+"/validation")
        test_summary_writer = SummaryWriter(log_dir+"/test")
    
    #path where the checkpoints will be saved
    checkpoint_filepath=os.path.join(path_experiment, 'checkpoint')
    #Loading filters
    num_filters=args.bwe.lpf.num_random_filters
    filters=lowpass_utils.get_random_FIR_filters(num_filters, mean_fc=args.bwe.lpf.mean_fc, std_fc=args.bwe.lpf.std_fc, device=device,sr=args.fs)# more parameters here

    loss_l1 = torch.nn.L1Loss()
    loss_l2 = torch.nn.MSELoss()

    optim_g = torch.optim.AdamW(gener_model.parameters(), args.bwe.lr_gen_l1, betas=[args.beta1,args.beta2])
    optim_d = torch.optim.AdamW(msd.parameters(), args.bwe.lr_dis_l2,betas=[args.bwe.discriminators.adamD.beta1, args.bwe.discriminators.adamD.beta2])

    training_stage=0
    
    for epoch in range(args.epochs):

        if epoch==args.bwe.epochs_in_stage_l1 and args.bwe.lambda_adv >0:
            training_stage=1
            #change lr
            for g in optim_g.param_groups:
                g['lr'] = args.bwe.lr_gen_l2

        train_loss=0
        step_loss=0
        
        
        for step in tqdm(range(int(args.steps_per_epoch)), desc="Training epoch "+str(epoch)):
            #Load noise and data
            optim_g.zero_grad()

            y=iterator.next()    
            y=y.to(device)

            y_lpf=lowpass_utils.apply_low_pass(y,filters,args) 
            x=y_lpf

            if args.bwe.add_noise.add_noise:
                #Adding noise regularization
                n=args.bwe.add_noise.power*torch.randn(y_lpf.shape)
                x=x+n.to(device) 

            y=y.unsqueeze(1)
            
            #APPLY GENERATOR
            if args.bwe.generator.variant=="unet2d": #Spectrogram domain generator
                xF =utils.do_stft(x, win_size=args.stft.win_size, hop_size=args.stft.hop_size, device=device)

                y_gF = gener_model(xF)

                y_g=utils.do_istft(y_gF, args.stft.win_size, args.stft.hop_size, device)

                y_g=y_g[:,0:x.shape[-1]]
                y_g=y_g.unsqueeze(1)

            else: #time domain generator
                y_g = gener_model(x)

            if training_stage==1:
                for i in range(args.bwe.repeat_disc_training):
                    optim_d.zero_grad()
                    #TRAIN DISCRIMINATORS REPEAT SEVERAL (2) TIMES
                    loss_disc_all=0
            
                    y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g.detach())   
                    loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminators.discriminator_loss(y_ds_hat_r, y_ds_hat_g) 

                    loss_disc_all += loss_disc_s
                    train_summary_writer.add_scalar('loss_disc_s', loss_disc_s, int(step+epoch*(args.steps_per_epoch)))

                    for k,(r, g) in enumerate(zip(losses_disc_s_r, losses_disc_s_g)):
                        train_summary_writer.add_scalar('loss_disc_s_'+str(k)+'_r', r, int(step+epoch*(args.steps_per_epoch)))
                        train_summary_writer.add_scalar('loss_disc_s_'+str(k)+'_g', g, int(step+epoch*(args.steps_per_epoch)))
                   
                    loss_disc_all.backward()
                    optim_d.step()
        
            loss_gen_all=0
    
            if args.bwe.lambda_freq_loss>0:
                stft_l1_loss, stft_sc_loss = stft_loss(y_g.squeeze(1), y.squeeze(1)) #check implementation #check why the squeezes
                loss_gen_all+=args.bwe.lambda_freq_loss*(stft_l1_loss + stft_sc_loss)
                
                train_summary_writer.add_scalar('loss_gen_stft', stft_l1_loss, int(step+epoch*(args.steps_per_epoch)))

            if args.bwe.use_mse_loss:
                loss_gen_all=loss_l2(y_g,y)
    
            if training_stage==1:
    
                y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g)
                loss_gen_s, losses_gen_s = unet2d_generator.generator_loss(y_ds_hat_g)

                loss_gen_all += args.bwe.lambda_adv * (loss_gen_s)    
                train_summary_writer.add_scalar('loss_gen_s', loss_gen_s, int(step+epoch*(args.steps_per_epoch)))
                for k, g in enumerate(losses_gen_s):
                    train_summary_writer.add_scalar('loss_gen_s_'+str(k), g, int(step+epoch*(args.steps_per_epoch)))
            train_summary_writer.add_scalar('loss_gen_all', loss_gen_all, int(step+epoch*(args.steps_per_epoch)))
    
                
            loss_gen_all.backward()
            optim_g.step()



        print("VALIDATION")

        loss_sample_val=0
        loss_stft_val=0
        SNR_val=0
        LSD_val=0
        if epoch%args.freq_inference==0:
            tb_audios=True
        else:
            tb_audios=False
        count=0
        
        for y in tqdm(val_loader):
            y=y.to(device)

            y_lpf=lowpass_utils.apply_low_pass(y,filters,args) 
            x=y_lpf
          
            if args.bwe.add_noise.add_noise:
                n=args.bwe.add_noise.power*torch.randn(y_lpf.shape)
                x=x+n.to(device) #not tested, need to tune the noise power
          
          
            #x=x.unsqueeze(1)
            y=y.unsqueeze(1)

            if args.bwe.generator.variant=="unet2d":
               
                xF =utils.do_stft(x, win_size=args.stft.win_size, hop_size=args.stft.hop_size, device=device)

                with torch.no_grad():
                    y_gF = gener_model(xF)

                y_g=utils.do_istft(y_gF, args.stft.win_size, args.stft.hop_size, device)
                y_g=y_g[:,0:x.shape[-1]]
                y_g=y_g.unsqueeze(1)
            else: #time-domain disc
                with torch.no_grad():
                    y_g = gener_model(x)

            count=count+1
            if tb_audios and count%int(len(val_loader.dataset)/(args.num_val_audios*args.batch_size_val))==0:
                #if count>5:
                #    tb_audios=False
                nf=torch.max(y)
                if epoch==0:
                    val_summary_writer.add_audio("noisy"+str(count), 0.9*x[0,:], epoch, sample_rate=args.fs)
                    val_summary_writer.add_image("noisy_spec_"+str(count), utils.generate_images(x[0,:]), epoch, dataformats='WHC')
                    val_summary_writer.add_audio("filtered"+str(count), 0.9*y_lpf[0,:], epoch, sample_rate=args.fs)
                    val_summary_writer.add_image("filtered_spec_"+str(count), utils.generate_images(y_lpf[0,:]), epoch,dataformats='WHC')
                    val_summary_writer.add_audio("clean"+str(count), 0.9*y[0,0,:], epoch, sample_rate=args.fs)
                    val_summary_writer.add_image("clean_spec_"+str(count), utils.generate_images(y[0,0,:]), epoch,dataformats='WHC')
                val_summary_writer.add_audio("bwe"+str(count), 0.9*y_g[0,0,:], epoch, sample_rate=args.fs)
                val_summary_writer.add_image("bwe_spec_"+str(count), utils.generate_images(y_g[0,0,:]), epoch, dataformats='WHC')

            stft_loss_l1, stft_loss_sc= stft_loss(y_g.squeeze(1), y.squeeze(1)) #check implementation #check why the squeezes
            loss_stft_val+=(stft_loss_l1 +stft_loss_sc)
            
            loss_sample_val += loss_l1(y, y_g) #import F, maybe torch.nn.l1loss? here
            
            SNR_val+= utils.computeSNR(y.squeeze(1),y_g.squeeze(1))
            LSD_val+= utils.computeLSD(y.squeeze(1),y_g.squeeze(1))

        loss_stft_val/=len(val_loader.dataset)
        loss_sample_val/=len(val_loader.dataset)
        SNR_val/=len(val_loader.dataset)
        LSD_val/=len(val_loader.dataset)

        print("SNR", SNR_val)
        print("LSD", LSD_val)
        val_summary_writer.add_scalar("val_loss_freq",loss_stft_val,epoch)
        val_summary_writer.add_scalar("val_loss_sample",loss_sample_val,epoch)
        val_summary_writer.add_scalar("val_SNR",SNR_val,epoch)
        val_summary_writer.add_scalar("val_LSD",LSD_val,epoch)

         
        if (epoch+1) % args.freq_save == 0:
            print("saving:_",checkpoint_filepath)
            torch.save(gener_model.state_dict(), checkpoint_filepath+"_"+str(epoch))

        if tb_audios:
            print("inferencing real (denoised) recordings")
            count=0
            for x in tqdm(test_loader):
                x=x.to(device)
                x_init=x

                if args.bwe.add_noise.add_noise:
                    n=args.bwe.add_noise.power*torch.randn(x.shape)
                    x=x+n.to(device) #not tested, need to tune the noise power

    
                if args.bwe.generator.variant=="unet2d":
                    xF =utils.do_stft(x,win_size=args.stft.win_size, hop_size=args.stft.hop_size, device=device)


                    with torch.no_grad():
                        y_gF = gener_model(xF)
                    
                    y_g=utils.do_istft(y_gF, args.stft.win_size, args.stft.hop_size, device)
                    y_g=y_g[:,0:x.shape[-1]]
                    y_g=y_g.unsqueeze(1)
                else:
                    with torch.no_grad():
                        y_g = gener_model(x)

                if epoch==0:
                    test_summary_writer.add_audio("real_original_"+str(count), 0.9*x_init[0,:], epoch, sample_rate=args.fs)
                    test_summary_writer.add_image("real_original_spec_"+str(count), utils.generate_images(x_init[0,:]), epoch, dataformats='WHC')

                    test_summary_writer.add_audio("real_noisy"+str(count), 0.9*x[0,:], epoch, sample_rate=args.fs)
                    test_summary_writer.add_image("real_noisy_spec_"+str(count), utils.generate_images(x[0,:]), epoch, dataformats='WHC')
                test_summary_writer.add_audio("real_bwe"+str(count), 0.9*y_g[0,0,:], epoch, sample_rate=args.fs)
                test_summary_writer.add_image("real_bwe_spec_"+str(count), utils.generate_images(y_g[0,0,:]), epoch, dataformats='WHC')
                count=count+1


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







