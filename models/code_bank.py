# Wentao Qiu, 2023-10-07
# qiuwentao1212@gmail.com

import torch
from torch import nn
from torch.nn import functional as F

from models.common import Conv1DBlock

class Attention(nn.Module):
    def __init__(self, channels: int, radius: int = 50, heads: int = 4):
        super().__init__()
        assert channels % heads == 0
        self.content = nn.Conv1d(channels, channels, 1)
        self.query = nn.Conv1d(channels, channels, 1)
        self.key = nn.Conv1d(channels, channels, 1)
        self.embedding = nn.Embedding(radius * 2 + 1, channels // heads)
        # Let's make this embedding a bit smoother
        weight = self.embedding.weight.data
        weight[:] = weight.cumsum(0) / torch.arange(1, len(weight) + 1).float().view(-1, 1).sqrt()
        self.heads = heads
        self.radius = radius

        self.bn = nn.BatchNorm1d(channels)
        self.fc = nn.Conv1d(channels, channels, 1)
        self.scale = nn.Parameter(torch.full([channels], 0.1))

    def forward(self, x):
        def _split(y):
            return y.view(y.shape[0], self.heads, -1, y.shape[2])

        content = _split(self.content(x))
        query = _split(self.query(x))
        key = _split(self.key(x))

        batch_size, _, dim, length = content.shape

        # first index `t` is query, second index `s` is key.
        dots = torch.einsum("bhct,bhcs->bhts", query, key)

        steps = torch.arange(length, device=x.device)
        relative = (steps[:, None] - steps[None, :])
        embs = self.embedding.weight.gather(
            0, self.radius + relative.clamp_(-self.radius, self.radius).view(-1, 1).expand(-1, dim))
        embs = embs.view(length, length, -1)
        dots += 0.3 * torch.einsum("bhct,tsc->bhts", query, embs)

        # we kill any reference outside of the radius
        dots = torch.where(
            relative.abs() <= self.radius, dots, torch.tensor(-float('inf')).to(embs))

        weights = torch.softmax(dots, dim=-1)
        out = torch.einsum("bhts,bhcs->bhct", weights, content)
        out += 0.3 * torch.einsum("bhts,tsc->bhct", weights, embs)
        out = out.reshape(batch_size, -1, length)
        out = F.relu(self.bn(self.fc(out))) * self.scale.view(1, -1, 1)
        return out

class SpatioTemporalCNN(nn.Module):
    def __init__(self, T, C):
        super().__init__()
        self.T = T
        self.C = C
        # Assuming the input shape is [bsz, T, C]
        self.spatial_conv1 = nn.Conv1d(in_channels=self.C, out_channels=48, kernel_size=3, padding=1)
        self.ln1 = nn.LayerNorm([48, T])
        self.time_conv1 = nn.Conv1d(in_channels=self.T, out_channels=40, kernel_size=3, padding=1)
        self.ln2 = nn.LayerNorm([40, 48])
        self.spatial_conv2 = nn.Conv1d(in_channels=48, out_channels=24, kernel_size=3, padding=1)
        self.ln3 = nn.LayerNorm([24, 40])
        self.time_conv2 = nn.Conv1d(in_channels=40, out_channels=20, kernel_size=3, padding=1)
        self.ln4 = nn.LayerNorm([20, 24])
        self.fc = nn.Linear(20 * 24, 128)

    def forward(self, x):
        x = x.transpose(1, 2)  # Shape: [bsz, C, T]
        x = self.spatial_conv1(x)
        x = self.ln1(x)
        x = F.gelu(x)
        x = x.transpose(1, 2)  # Shape: [bsz, T, C]
        x = self.time_conv1(x)
        x = self.ln2(x)
        x = F.gelu(x)
        x = x.transpose(1, 2)
        x = self.spatial_conv2(x)
        x = self.ln3(x)
        x = F.gelu(x)
        x = x.transpose(1, 2)
        x = self.time_conv2(x)
        x = self.ln4(x)
        x = F.gelu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
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
        x = self.ln0(x)
        x = F.gelu(x) 
        x = self.time_deconv1(x)
        x = self.ln1(x)
        x = F.gelu(x)
        x = x.transpose(1, 2)
        x = self.spatial_deconv1(x)
        x = self.ln2(x)
        x = F.gelu(x)
        x = x.transpose(1, 2)
        x = self.time_deconv2(x)
        x = x.transpose(1, 2)
        x = self.spatial_deconv2(x) 
        x = x.transpose(1, 2)
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


class defossez_encoder(nn.Module):
    def __init__(self,in_channel,out_channel,n_time):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        # self.conv1d_block = nn.Sequential(
        #     Conv1DBlock(self.in_channel,[self.in_channel, self.in_channel, self.in_channel*2], [3, 3, 3], [1, 2, 2], n_time),
        #     Conv1DBlock(self.in_channel,[self.in_channel, self.in_channel, self.in_channel*2], [3, 3, 3], [4, 8, 2], n_time),
        #     Conv1DBlock(self.in_channel,[self.in_channel, self.in_channel, self.in_channel*2], [3, 3, 3], [16, 1, 2], n_time),
        #     Conv1DBlock(self.in_channel,[self.in_channel, self.in_channel, self.in_channel*2], [3, 3, 3], [2, 4, 2], n_time),
        #     Conv1DBlock(self.in_channel,[self.in_channel, self.in_channel, self.in_channel*2], [3, 3, 3], [8, 16, 2], n_time),
        # )
        self.conv1d_block = nn.Sequential(
            Conv1DBlock(self.in_channel,[self.in_channel, self.in_channel, self.in_channel*2], [3, 3, 3], [1, 2, 2], n_time),
            Conv1DBlock(self.in_channel,[self.in_channel, self.in_channel, self.in_channel*2], [3, 3, 3], [4, 8, 2], n_time),
        )

        self.feature_block = nn.Sequential(
            nn.Conv1d(self.in_channel, self.in_channel*2, kernel_size=1),
            nn.LayerNorm([self.in_channel*2, n_time]),
            nn.GELU(),
            nn.Conv1d(self.in_channel*2, self.out_channel, kernel_size=1)
        )

    def forward(self, inputs):
        outputs = self.conv1d_block(inputs)
        outputs = self.feature_block(outputs.permute(0, 2, 1)) # (batch_size, n_features, seq_len)
        outputs = outputs.permute(0, 2, 1)
        return outputs # (batch_size, seq_len, n_features)

### Attention
"""
class SpatioTemporalCNN_attention(nn.Module):
    def __init__(self, T, attention_radius=15, attention_heads=4):
        super(SpatioTemporalCNN_attention, self).__init__()
        self.T = T
        # Group Convolution Layers
        self.group_conv1 = nn.Conv1d(in_channels=384, out_channels=192, kernel_size=3, groups=4, padding=1)
        self.ln1 = nn.LayerNorm([192, T])
        self.group_conv2 = nn.Conv1d(in_channels=192, out_channels=64, kernel_size=3, groups=4, padding=1)
        self.ln2 = nn.LayerNorm([64, T])

        # Attention Layer for Time Dimension
        self.time_attention = Attention(channels=64, radius=attention_radius, heads=attention_heads)
        self.ln3 = nn.LayerNorm([64, T])
        # fully connected layer, dimension reduction
        self.fc_time_reduction1 = nn.Linear(self.T, 50)
        self.fc_time_reduction2 = nn.Linear(50, 30)
        '''
        maybe I also need fc for spatial dimension reduction, wait for a try
        '''

    def forward(self, x):
        layer_outputs = {}
        layer_outputs['input'] = x
        # Reshape the input for group convolution layers
        x = x.permute(0, 2, 1)  # Shape: [bsz, C, T]
        # Apply group convolution layers
        x = F.gelu(self.ln1(self.group_conv1(x)))
        layer_outputs['group_conv1'] = x.permute(0, 2, 1)
        x = F.gelu(self.ln2(self.group_conv2(x)))
        layer_outputs['group_conv2'] = x.permute(0, 2, 1)
        x = self.time_attention(x)
        x = F.gelu(self.ln3(x))
        layer_outputs['time_attention'] = x.permute(0, 2, 1)
        x = self.fc_time_reduction1(x)
        x = F.gelu(x)
        layer_outputs['fc_time_reduction1'] = x.permute(0, 2, 1)
        x = self.fc_time_reduction2(x)
        x = F.gelu(x)  # Shape: [bsz, C, T]
        layer_outputs['fc_time_reduction2'] = x.permute(0, 2, 1)
        return x, layer_outputs
        # return x

class Decoder_SpatioTemporalCNN_Attention(nn.Module):
    def __init__(self, T, attention_radius=15, attention_heads=4):
        super(Decoder_SpatioTemporalCNN_Attention, self).__init__()
        self.T = T
        self.fc_time_expansion1 = nn.Linear(30, 50)
        self.fc_time_expansion2 = nn.Linear(50, self.T)
        self.time_attention = Attention(channels=64, radius=attention_radius, heads=attention_heads)
        self.ln2 = nn.LayerNorm([64, T])
        self.group_deconv2 = nn.ConvTranspose1d(in_channels=64, out_channels=192, kernel_size=3, groups=4, padding=1)
        self.ln1 = nn.LayerNorm([192, T])
        self.group_deconv1 = nn.ConvTranspose1d(in_channels=192, out_channels=384, kernel_size=3, groups=4, padding=1)

    def forward(self, x):
        x = F.gelu(self.fc_time_expansion1(x))
        x = F.gelu(self.fc_time_expansion2(x))
        x = self.time_attention(x)
        x = F.gelu(self.ln2(x))
        x = F.gelu(self.ln1(self.group_deconv2(x)))
        x = self.group_deconv1(x))
        x = x.permute(0, 2, 1)  # Shape: [bsz, T, C]
        return x

class SpatioTemporalAutoEncoder(nn.Module):
    def __init__(self, T, attention_radius=15, attention_heads=4):
        super(SpatioTemporalAutoEncoder, self).__init__()
        self.encoder = SpatioTemporalCNN_attention(T, attention_radius, attention_heads)
        self.decoder = Decoder_SpatioTemporalCNN_Attention(T, attention_radius, attention_heads)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
"""

if __name__ == '__main__':
    
    # Create a random input tensor.
    inputs = torch.randn(5, 82, 96)
    defossez_encoder = defossez_encoder(96, 128, 82)
    outputs = defossez_encoder(inputs)
    print(outputs.shape)
