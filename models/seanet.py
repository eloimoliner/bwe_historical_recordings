import torch.nn as nn
import math as m
import torch
#import torchaudio

#PyTorch implementation of the SEANet model https://arxiv.org/abs/2010.10677. The implementation might differ from the original one.


class Decoder(nn.Module):
    '''
    [B, T, F, N] , skip connections => [B, T, F, N]  
    Decoder side of the U-Net subnetwork.
    '''
    def __init__(self, Ns, Ss, Ks):
        super(Decoder, self).__init__()

        self.Ns=Ns
        self.Ks=Ks
        self.depth=len(Ss)

        self.final_conv=nn.Conv1d(128,1,kernel_size=7, padding="same")
        self.layers=nn.ModuleList()
        self.upsamplers=nn.ModuleList()
        N0=512
        self.cropconcat = CropConcatBlock()
        for i in range(self.depth):
            if i==(self.depth-1):
                Nin=N0
            else:
                Nin=self.Ns[i+1]
            if i==0:
                Nout=128
            else:
                Nout=self.Ns[i]
            self.upsamplers.append(
                nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.ConvTranspose1d(
                    Nin,
                    Nout,
                    kernel_size=self.Ks[i],
                    stride=Ss[i],
                    padding=Ss[i] // 2 + Ss[i] % 2,
                    output_padding=Ss[i] % 2,
                )))
            self.layers.append(
                nn.Sequential(
                ResnetBlock(Nout*2,Nout, dilation=1),
                ResnetBlock(Nout,Nout, dilation=3),
                ResnetBlock(Nout,Nout, dilation=9)))


    def forward(self,inputs, contracting_layers):
        x=inputs
        for i in range(self.depth,0,-1):
            x=self.upsamplers[i-1](x)
            x=self.cropconcat(x,contracting_layers[i-1])
            x=self.layers[i-1](x)
        x=self.final_conv(x)
        return x 


class Encoder(nn.Module):

    '''
    [B, T, F, N] => skip connections , [B, T, F, N_4]  
    Encoder side of the U-Net subnetwork.
    '''
    def __init__(self, Ns,Ss, Ks):
        super(Encoder, self).__init__()
        self.Ns=Ns
        self.Ks=Ks
        self.depth=len(Ss)

        self.contracting_layers = {}

        self.layers=nn.ModuleList()
        self.downsamplers=nn.ModuleList()
        N0=1
        self.first_conv=nn.Conv1d(N0,64,kernel_size=7, padding="same")
        for i in range(self.depth):
            if i==0:
                Nin=64
            else:
                Nin=self.Ns[i-1]
            Nout=self.Ns[i]

            self.layers.append(nn.Sequential(
                ResnetBlock(Nin,Nout, dilation=1),
                ResnetBlock(Nout,Nout, dilation=3),
                ResnetBlock(Nout,Nout, dilation=9),
                ResnetBlock(Nout,Nout, dilation=27)))

            self.downsamplers.append(nn.Sequential(
                nn.Conv1d(
                    Nout,
                    Nout,
                    kernel_size=self.Ks[i],
                    stride=Ss[i],
                    padding=(self.Ks[i])//2,
                ),
                nn.LeakyReLU(0.2)))


    def forward(self, inputs):
        x=inputs
        x=self.first_conv(x)

        for i in range(self.depth):

            x =self.layers[i](x)
        
            self.contracting_layers[i] = x 
            
            x=self.downsamplers[i](x)

        return x, self.contracting_layers

class ResnetBlock(nn.Module):
    def __init__(self,N0, dim, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(dilation),
            nn.Conv1d(N0, dim, kernel_size=3, dilation=dilation),
            nn.LeakyReLU(0.2),
            nn.Conv1d(dim, dim, kernel_size=1),
        )
        self.shortcut = nn.Conv1d(N0, dim, kernel_size=1)

    def forward(self, x):
        return self.shortcut(x) + self.block(x)

class Unet1d(nn.Module):
    #time-domain U-Net sor audio superresolution, like kuleshov et al.

    def __init__(self):
        super(Unet1d, self).__init__()
        #self.L=unet_args.depth

        #n_filters = [128, 384, 512, 512, 512, 512, 512, 512]
        n_filters = [128, 128, 256 ,512, 512]
        

        Ss= [2,2,2,4,4]
        n_filtersizes = [16,16,16,16,16]

                        
        self.encoder=Encoder(n_filters, Ss, n_filtersizes)
        Nout=512 

        self.bottleneck=nn.Sequential(
                                nn.Conv1d(512,128,kernel_size=16,padding="same"),
                                nn.Conv1d(128,512,kernel_size=16, padding="same"))

        self.decoder=Decoder(n_filters, Ss, n_filtersizes)


        self.cropconcat = CropConcatBlock()


    def forward(self, inputs):
        inputs=inputs.unsqueeze(1)
        x, contracting_layers= self.encoder(inputs)
        
        #decoder
        x=self.bottleneck(x)

        xout =self.decoder(x, contracting_layers) #None, None, 1025, 32 features

        xout=torch.add(xout, inputs)

        return xout
            


class CropConcatBlock(nn.Module):

    def forward(self, down_layer, x, **kwargs):
        x1_shape = down_layer.shape
        x2_shape = x.shape

        height_diff = (x1_shape[2] - x2_shape[2]) // 2
        down_layer_cropped = down_layer[:,
                                        :,
                                        height_diff: (x2_shape[2] + height_diff)]
        x = torch.cat((down_layer_cropped, x),1)
        return x
