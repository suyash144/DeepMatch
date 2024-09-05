# Wentao Qiu, 2023-10-07
# qiuwentao1212@gmail.com

import torch
from torch import nn
from torch.nn import functional as F


if __name__ == '__main__':
    from common import Conv1DSeq, Deconv1DSeq
else:
    from models.common import Conv1DSeq, Deconv1DSeq


class SpatioTemporalCNN(nn.Module):
    def __init__(self, n_channel,n_time,n_output=256):
        super().__init__()
        self.n_channel = n_channel
        self.n_time = n_time
        self.n_output = n_output
        # Assuming the input shape is [bsz, T, C]
        self.spatial_conv1 = nn.Conv1d(in_channels=self.n_channel, out_channels=self.n_channel, kernel_size=3, padding=1)
        self.ln1 = nn.LayerNorm([self.n_channel, self.n_time])
        self.time_conv1 = nn.Conv1d(in_channels=self.n_time, out_channels=40, kernel_size=3, padding=1)
        self.ln2 = nn.LayerNorm([40, 48])
        self.spatial_conv2 = nn.Conv1d(in_channels=48, out_channels=24, kernel_size=3, padding=1)
        self.ln3 = nn.LayerNorm([24, 40])
        self.time_conv2 = nn.Conv1d(in_channels=40, out_channels=20, kernel_size=3, padding=1)
        self.ln4 = nn.LayerNorm([20, 24])
        self.fc = nn.Linear(20 * 24, 128)

    def forward(self, x):
        """
        :inputs: (batch_size, n_time, n_channels) 
        :return outputs: (batch_size, self.n_output)
        """
        x = x.transpose(1, 2)  # Shape: [bsz, C, T]
        x = F.gelu(self.ln1(self.spatial_conv1(x)))
        x = x.transpose(1, 2)  # Shape: [bsz, T, C]
        x = F.gelu(self.ln2(self.time_conv1(x)))
        x = x.transpose(1, 2)
        x = F.gelu(self.ln3(self.spatial_conv2(x)))
        x = x.transpose(1, 2)
        x = F.gelu(self.ln4(self.time_conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SpatioTemporalCNN_V2(nn.Module):
    def __init__(self,n_channel,n_time,n_output=256):
        super().__init__()
        self.n_channel = n_channel
        self.n_time = n_time
        self.n_output = n_output
        self.ConvBlock1 = Conv1DSeq(self.n_channel, [self.n_channel, self.n_channel],[3,3],[1,2],self.n_time)
        self.ConvBlock2 = Conv1DSeq(self.n_time, [self.n_time, self.n_time],[3,3],[1,2],self.n_channel)
        
        self.n_channel_red = self.n_channel//2
        self.SpatialBlock = nn.Sequential(
            nn.Conv1d(self.n_channel,self.n_channel_red, kernel_size=1),
            nn.LayerNorm([self.n_channel_red, self.n_time]),
            nn.GELU(),
        )

        self.FcBlock = nn.Sequential(
            nn.Linear(self.n_channel_red*self.n_time, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, self.n_output),
        )

        self.__init_weight()
    
    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
    def forward(self, x):
        """
        :inputs: (batch_size, n_time, n_channels) 
        :return outputs: (batch_size, self.n_output)
        """
        x = self.ConvBlock1(x) # after this, the shape is [bsz, C, T]
        x = self.ConvBlock2(x) # after this, the shape is [bsz, T, C]
        x = x.permute(0,2,1)
        x = self.SpatialBlock(x) # after this, the shape is [bsz, C_red, T]
        x = x.view(x.shape[0],-1)
        x = self.FcBlock(x)
        return x
    
class SpatioTemporalCNN_V3(nn.Module):
    def __init__(self,n_channel,n_time,n_output=256):
        super().__init__()
        self.n_channel = n_channel
        self.n_time = n_time
        self.n_output = n_output
        self.ConvBlock1 = Conv1DSeq(self.n_channel, [self.n_channel, self.n_channel],[3,3],[1,2],self.n_time)
        self.n_channel_red = self.n_channel//2
        self.SpatialBlock = nn.Sequential(
            nn.Conv1d(self.n_channel,self.n_channel_red, kernel_size=1),
            nn.LayerNorm([self.n_channel_red, self.n_time]),
            nn.GELU(),
        )
        self.FcBlock = nn.Sequential(
            nn.Linear(self.n_channel_red*self.n_time, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, self.n_output),
        )

        self.__init_weight()
    
    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
    def forward(self, x):
        """
        :inputs: (batch_size, n_time, n_channels) 
        :return outputs: (batch_size, self.n_output)
        """
        x = self.ConvBlock1(x) # after this, the shape is [bsz, C, T]
        x = self.SpatialBlock(x) # after this, the shape is [bsz, C_red, T]
        x = x.view(x.shape[0],-1)
        x = self.FcBlock(x)
        return x
    
class Decoder_SpatioTemporalCNN(nn.Module):
    def __init__(self, T, C):
        super().__init__()
        self.T = T
        self.C = C
        self.fc = nn.Linear(128, 20 * 24)
        self.ln0 = nn.LayerNorm([20, 24])
        # Layer definitions should be the reverse of the encoder
        self.time_deconv1 = nn.ConvTranspose1d(in_channels=20, out_channels=40, kernel_size=3, padding=1)
        self.ln1 = nn.LayerNorm([40, 24])
        self.spatial_deconv1 = nn.ConvTranspose1d(in_channels=24, out_channels=48, kernel_size=3, padding=1)
        self.ln2 = nn.LayerNorm([48, 40])
        self.time_deconv2 = nn.ConvTranspose1d(in_channels=40, out_channels=self.T, kernel_size=3, padding=1)
        self.ln3 = nn.LayerNorm([self.T, 48])
        self.spatial_deconv2 = nn.ConvTranspose1d(in_channels=48, out_channels=self.C, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 20, 24)
        x = F.gelu(self.ln0(x)) 
        x = F.gelu(self.ln1(self.time_deconv1(x)))
        x = x.transpose(1, 2)
        x = F.gelu(self.ln2(self.spatial_deconv1(x)))
        x = x.transpose(1, 2)
        x = self.time_deconv2(x)
        x = x.transpose(1, 2)
        x = self.spatial_deconv2(x) 
        x = x.transpose(1, 2)
        return x


class Decoder_SpatioTemporalCNN_V2(nn.Module):
    def __init__(self, n_channel,n_time,n_input=256):
        super().__init__()
        self.n_channel = n_channel
        self.n_time = n_time
        self.n_channel_red = self.n_channel//2
        self.FcBlock = nn.Sequential(
            nn.Linear(n_input, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, self.n_channel_red*self.n_time),
        )
        self.DeConvSpatialBlock = nn.Sequential(
            nn.ConvTranspose1d(self.n_channel_red, self.n_channel, kernel_size=1),
            nn.LayerNorm([self.n_channel, self.n_time]),
            nn.GELU(),
        )
        self.DeConvSpatialBlock2 = nn.Sequential(
            nn.ConvTranspose1d(self.n_channel, self.n_channel, kernel_size=3, padding=1),
            nn.LayerNorm([self.n_channel, self.n_time]),
            nn.GELU(),
        )
        self.DeConvTimeLayer = nn.ConvTranspose1d(self.n_time, self.n_time, kernel_size=3, padding=1)
        self.__init_weight()
    
    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        '''
        Inputs: (batch_size, n_input)
        Outputs: (batch_size, n_time, n_channel)
        '''
        x = self.FcBlock(x)
        x = x.view(x.shape[0],self.n_channel_red,self.n_time)
        x = self.DeConvSpatialBlock(x) # after this, the shape is [bsz, C, T]
        x = self.DeConvSpatialBlock2(x)
        x = x.permute(0,2,1)
        x = self.DeConvTimeLayer(x)
        return x

class Decoder_SpatioTemporalCNN_V3(nn.Module):
    def __init__(self, n_channel,n_time,n_input=256):
        super().__init__()
        self.n_channel = n_channel
        self.n_time = n_time
        self.n_channel_red = self.n_channel//2
        self.FcBlock = nn.Sequential(
            nn.Linear(n_input, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, self.n_channel_red*self.n_time),
        )
        self.DeConvSpatialBlock = nn.Sequential(
            nn.ConvTranspose1d(self.n_channel_red, self.n_channel, kernel_size=1),
            nn.LayerNorm([self.n_channel, self.n_time]),
            nn.GELU(),
        )
        self.DeConvBlock1 = Deconv1DSeq(self.n_channel,[self.n_channel, self.n_channel], [3, 3], [2, 1],self.n_time)
        self.__init_weight()
    
    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        '''
        Inputs: (batch_size, n_input)
        Outputs: (batch_size, n_time, n_channel)
        '''
        x = self.FcBlock(x)
        x = x.view(x.shape[0],self.n_channel_red,self.n_time)
        x = self.DeConvSpatialBlock(x)
        x = self.DeConvBlock1(x)
        return x
    
class SpatioTemporalAutoEncoder(nn.Module):
    def __init__(self, T, C):
        super().__init__()
        self.T = T
        self.C = C
        self.encoder = SpatioTemporalCNN(self.T, self.C)
        self.decoder = Decoder_SpatioTemporalCNN(self.T, self.C)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class SpatioTemporalAutoEncoder_V2(nn.Module):
    def __init__(self,n_channel,n_time,n_output=256):
        super().__init__()
        self.n_channel = n_channel
        self.n_time = n_time
        self.n_output = n_output
        self.encoder = SpatioTemporalCNN_V2(self.n_channel,self.n_time,self.n_output)
        self.decoder = Decoder_SpatioTemporalCNN_V2(self.n_channel,self.n_time,self.n_output)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
class SpatioTemporalAutoEncoder_V3(nn.Module):
    def __init__(self,n_channel,n_time,n_output=256):
        super().__init__()
        self.n_channel = n_channel
        self.n_time = n_time
        self.n_output = n_output
        self.encoder = SpatioTemporalCNN_V3(self.n_channel,self.n_time,self.n_output)
        self.decoder = Decoder_SpatioTemporalCNN_V3(self.n_channel,self.n_time,self.n_output)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
if __name__ == '__main__':
    # Create a random input tensor.
    inputs = torch.randn(5, 60, 30)
    # encoder = SpatioTemporalCNN_V2(30, 60, 256)
    # decoder = Decoder_SpatioTemporalCNN_V2(30, 60, 256)
    # outputs = encoder(inputs)
    # print(outputs.shape)
    # decoded = decoder(outputs)
    # print(decoded.shape)

    AE_V2 = SpatioTemporalAutoEncoder_V2(30, 60, 256)
    decoded = AE_V2(inputs)
    print(decoded.shape)
