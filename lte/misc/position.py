import torch
import torch.nn as nn
import math
from torch import Tensor


class PositionalEncoding(nn.Module):
    """ slight modification from https://pytorch.org/tutorials/beginner/transformer_tutorial.html """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.register_buffer('pe', sinusoidal(d_model, max_len))

    def forward(self, x):
        """
        Args:
            x (Tensor): the input tensor of shape (seq_len, batch, embed)
        Returns:
            x (Tensor): the output tensor of shape (seq_len, batch, embed)
        """
        return x + self.pe[:x.size(0)]


def sinusoidal(d_model, max_len=5000):
    """
    Create a sinusoidal positional encoding.
    
    Args:
        d_model (int): the model dimension
        max_len (int): the maximum length of the sequence
    Returns:
        pe (Tensor): the positional encoding tensor of shape (1, max_len, d_model)
    """
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2)
                         * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, 1, d_model)
    pe[:, 0, 0::2] = torch.sin(position * div_term)
    pe[:, 0, 1::2] = torch.cos(position * div_term)
    return pe.permute(1, 0, 2)


def sinusoidal_2d(d_model, height, width, normalization_constant=None):
    """
    Create a 2D sinusoidal positional encoding.

    Args:
        d_model (int): the model dimension
        height (int): the height of the 2D grid
        width (int): the width of the 2D grid
    Returns:
        pe (Tensor): the positional encoding tensor of shape (1, height*width, d_model)
    """
    if normalization_constant is None:
        normalization_constant = height * width

    # calculate div_term for both dimensions
    # this controls for the frequency cos(w / div_term)
    # not sure what the choice of using exp(log()) is
    div_term = torch.exp(torch.arange(0, d_model, 2) *
                         (-math.log(normalization_constant) / d_model))

    # create position encoding for height
    pe_y = torch.zeros(height, d_model)
    pe_y[:, 0::2] = torch.sin(torch.arange(height).unsqueeze(1) * div_term)
    pe_y[:, 1::2] = torch.cos(torch.arange(height).unsqueeze(1) * div_term)
    pe_y = pe_y.unsqueeze(1).repeat(1, width, 1)  # Repeat for each width

    # create position encoding for width
    pe_x = torch.zeros(width, d_model)
    pe_x[:, 0::2] = torch.sin(torch.arange(width).unsqueeze(1) * div_term)
    pe_x[:, 1::2] = torch.cos(torch.arange(width).unsqueeze(1) * div_term)
    pe_x = pe_x.unsqueeze(0).repeat(height, 1, 1)  # Repeat for each height

    # combine the encodings by adding
    pe = pe_y + pe_x

    # reshape to match the expected output shape
    return pe.view(1, height * width, d_model)
