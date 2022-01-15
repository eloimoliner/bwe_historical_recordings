"""
U-Net model for 1-D signals
"""
import torch.nn as nn
import torch

class PixelUpscale(torch.nn.Module):
    """
    1D pixel shuffler. https://arxiv.org/pdf/1609.05158.pdf
    Upscales activations
    """
    def __init__(self, upscale_factor, odd_output=False):
        super(PixelUpscale, self).__init__()
        self.upscale_factor = upscale_factor
        self.odd_output = odd_output

    def forward(self, x):

        batch_size = x.shape[0]
        short_channel_len = x.shape[1]
        short_width = x.shape[2]

        long_channel_len = short_channel_len // self.upscale_factor
        long_width = self.upscale_factor * short_width

        x = x.contiguous().view([batch_size, self.upscale_factor, long_channel_len, short_width])
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, long_channel_len, long_width)
        if self.odd_output:
            x = x[:, :, :-1]

        return x

def conv1d_same(in_channels, out_channels, kernel_size, bias=True, dilation=1, pad_type='zero'):
    # Convolution which does not change input size
    if pad_type == 'zero':
        return torch.nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=(kernel_size - 1) // 2, bias=bias, dilation=dilation)
    elif pad_type == 'reflection':
        return torch.nn.Sequential(
            torch.nn.ReflectionPad1d((kernel_size - 1) // 2),
            torch.nn.Conv1d(in_channels, out_channels, kernel_size, bias=bias, dilation=dilation)                         
        )
    elif pad_type == 'replication':
        return torch.nn.Sequential(
            torch.nn.ReplicationPad1d((kernel_size - 1) // 2),
            torch.nn.Conv1d(in_channels, out_channels, kernel_size, bias=bias, dilation=dilation)                         
        )
    else:
        raise ValueError('Unknown padding type')

def conv1d_halve(in_channels, out_channels, kernel_size, bias=True, dilation=1, pad_type='zero'):
    # Convolution which halves the input size
    if pad_type == 'zero':
        return torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=2,
                               padding=(kernel_size - 1) // 2, bias=bias, dilation=dilation)
    elif pad_type == 'reflection':
        return torch.nn.Sequential(
            torch.nn.ReflectionPad1d((kernel_size - 1) // 2),
            torch.nn.Conv1d(in_channels, out_channels, kernel_size, bias=bias, dilation=dilation, stride=2)                         
        )
    elif pad_type == 'replication':
        return torch.nn.Sequential(
            torch.nn.ReplicationPad1d((kernel_size - 1) // 2),
            torch.nn.Conv1d(in_channels, out_channels, kernel_size, bias=bias, dilation=dilation, stride=2)                         
        )
    else:
        raise ValueError('Unknown padding type')


class Model(nn.Module):

    def __init__(self, mono=False, scale=2, odd_length=False,
                 pad_type='zero', dropout=0.0, batchnorm=False):
        super(Model, self).__init__()

        assert not (batchnorm and dropout)
        self.scale = scale

        # Mono/Stereo
        if mono:
            n_input_ch = 1
            n_output_ch = 1
        else:
            n_input_ch = 2
            n_output_ch = 2

        ch_down_net = [n_input_ch, 128, 256, 512, 512, 512]
        ker_down_net = [65, 33, 17, 9, 9]

        ch_down_bottle = 512
        ker_down_bottle = 9

        ch_up_bottle = 512
        ker_up_bottle = 9

        ch_up_net = [512, 512, 512, 256, 128, n_output_ch]
        ker_up_net = [17, 33, 65, 9,9]

        # note that activations from downsampling layers will be stacked onto them

        activation = nn.ReLU(inplace=True)

        self.down_net = nn.ModuleList()   # downsampling network

        for i in range(len(ker_down_net)):
            down_block = nn.ModuleList()

            down_block.append(conv1d_halve(ch_down_net[i], ch_down_net[i+1],
                                           ker_down_net[i], pad_type=pad_type))
            if batchnorm:
                down_block.append(nn.BatchNorm1d(ch_down_net[i+1]))
            down_block.append(activation)

            down_block = nn.Sequential(*down_block)
            self.down_net.append(down_block)

        # bottleneck doesn't have residual connections
        bottleneck = nn.ModuleList()
        # downsampling block
        bottleneck.append(conv1d_halve(ch_down_net[-1], ch_down_bottle, ker_down_bottle, pad_type=pad_type))
        if dropout > 0:
            bottleneck.append(nn.Dropout(dropout))
        bottleneck.append(activation)
        # upsampling block
        bottleneck.append(conv1d_same(ch_down_bottle, ch_up_bottle * scale, ker_up_bottle, pad_type=pad_type))
        if dropout > 0:
            bottleneck.append(nn.Dropout(dropout))
        bottleneck.append(activation)
        bottleneck.append(PixelUpscale(self.scale, odd_output=odd_length))

        self.bottleneck = nn.Sequential(*bottleneck)

        self.up_net = nn.ModuleList()   # upsampling network
        for i in range(len(ker_up_net)):

            n_ch_in = ch_up_net[i] * 2     # residual channels will be added
            n_ch_out = ch_up_net[i+1] * self.scale      # for pixel upscaling

            up_block = nn.ModuleList()
            up_block.append(conv1d_same(n_ch_in, n_ch_out, ker_up_net[i], pad_type=pad_type))
            if dropout > 0:
                up_block.append(nn.Dropout(dropout))
            if i < len(ker_up_net) - 1:     # no activation after last layer
                up_block.append(activation)
            up_block.append(PixelUpscale(self.scale, odd_output=odd_length))

            up_block = nn.Sequential(*up_block)
            self.up_net.append(up_block)
            self.cropconcat = CropConcatBlock()

    def forward(self, x):
        x=x.unsqueeze(1)
        y = x.clone()
        res = []    # keep downsampled activations to use in stacking residual connections to upsampling network
        for down_block in self.down_net:
            y = down_block(y)
            res.append(y)
    
        y = self.bottleneck(y)

        for i, up_block in enumerate(self.up_net):
            y = self.cropconcat(y, res[-i-1])   # concat along channel dimension
            y = up_block(y)

        y = y + x

        return y

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
