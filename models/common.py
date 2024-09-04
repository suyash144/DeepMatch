import torch
import torch.nn as nn
import torch.nn.functional as F

    
class Conv1DSeq(nn.Module):
    """
    Conv1DSeq layer used to convolve input data.
    """
    def __init__(self, n_input_channels, n_filters, kernel_sizes, dilation_rates, expected_seq_len):
        """
        Initialize `Conv1DSeq` object.
        :param n_filters: (2[list],) - The dimensionality of the output space.
        :param kernel_sizes: (2[list],) - The length of the 1D convolution window.
        :param dilation_rates: (2[list],) - The dilation rate to use for dilated convolution.
        :param expected_seq_len: int - The expected sequence length after convolution.
        """
        super(Conv1DSeq, self).__init__()

        assert len(n_filters) == len(kernel_sizes) == len(dilation_rates) == 2
        self.n_filters = n_filters
        self.kernel_sizes = kernel_sizes
        self.dilation_rates = dilation_rates

        # Initialize the first component of `Conv1DSeq`.
        self.conv1 = nn.Conv1d(n_input_channels, n_filters[0], kernel_sizes[0], padding="same",
                               dilation=dilation_rates[0])
        self.ln1 = nn.LayerNorm([n_filters[0],expected_seq_len])
        
        # Initialize the second component of `Conv1DSeq`.
        self.conv2 = nn.Conv1d(n_filters[0], n_filters[1], kernel_sizes[1], padding="same",
                               dilation=dilation_rates[1])
        self.ln2 = nn.LayerNorm([n_filters[1], expected_seq_len])
        self.gelu = nn.GELU()

    def forward(self, inputs):
        """
        :param inputs: (batch_size, seq_len, n_input_channels) - The input data.
        :return outputs: (batch_size, seq_len, n_output_channels) - The convolved data.
        """
        outputs = self.conv1(inputs.permute(0, 2, 1)) + inputs.permute(0, 2, 1)
        # outputs = nn.functional.gelu(self.ln1(outputs))
        outputs = self.gelu(self.ln1(outputs))
        outputs = self.conv2(outputs) + outputs
        # outputs = nn.functional.gelu(self.ln2(outputs))
        outputs = self.gelu(self.ln2(outputs))
        return outputs

class Deconv1DSeq(nn.Module):
    """
    Deconv1DSeq layer used to deconvolve input data.
    """
    def __init__(self, n_input_channels, n_filters, kernel_sizes, dilation_rates, expected_seq_len):
        """
        Initialize `Deconv1DSeq` object.
        """
        super(Deconv1DSeq, self).__init__()

        assert len(n_filters) == len(kernel_sizes) == len(dilation_rates) == 2
        self.n_filters = n_filters
        self.kernel_sizes = kernel_sizes
        self.dilation_rates = dilation_rates

        # Calculate padding for "same" output length. Adjust as needed.
        padding1 = ((expected_seq_len - 1) * 1 - expected_seq_len + dilation_rates[1] * (kernel_sizes[1] - 1) + 1) // 2
        padding2 = ((expected_seq_len - 1) * 1 - expected_seq_len + dilation_rates[0] * (kernel_sizes[0] - 1) + 1) // 2
        self.deconv1 = nn.ConvTranspose1d(n_filters[1], n_filters[0], kernel_sizes[1],
                                          padding=padding1, dilation=dilation_rates[1])
        self.ln1 = nn.LayerNorm([n_filters[0], expected_seq_len])
        self.deconv2 = nn.ConvTranspose1d(n_filters[0], n_input_channels, kernel_sizes[0],
                                          padding=padding2, dilation=dilation_rates[0])
        self.ln2 = nn.LayerNorm([n_input_channels, expected_seq_len])
        self.gelu = nn.GELU()

    def forward(self, inputs):
        """
        :param inputs: (batch_size, n_input_channels,seq_len) - The input data.
        :return outputs: (batch_size,seq_len, n_output_channels) - The deconvolved data.
        """
        outputs = self.deconv1(inputs) + inputs
        outputs = self.gelu(self.ln1(outputs))
        outputs = self.deconv2(outputs) + outputs
        # outputs = self.gelu(self.ln2(outputs)) # for decoder, we don't need to use gelu or layer norm
        return outputs.permute(0, 2, 1)


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
    
if __name__ == '__main__':
    # # Create a `Conv1DBlock` object.
    # conv1d_block = Conv1DBlock(n_input_channels=256, n_filters=[256, 256, 256],
    #                            kernel_sizes=[33, 39, 51], dilation_rates=[1, 3, 9],expected_seq_len=256)
    # # Create a random input tensor.
    # inputs = torch.randn(32, 256, 256)
    # # Get the outputs.
    # outputs = conv1d_block(inputs)
    # # Print the shape of `outputs`.
    # print(outputs.shape)

    # Create a `Deconv1DSeq` object.
    inputs = torch.randn(32, 46, 82) # (batch_size, n_input_channels, seq_len)
    deconv1d_seq = Deconv1DSeq(n_input_channels=46, n_filters=[46, 46], kernel_sizes=[3, 3],
                               dilation_rates=[1, 3], expected_seq_len=82)
    # Get the outputs.
    outputs = deconv1d_seq(inputs)
    # Print the shape of `outputs`.
    print(outputs.shape)