#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1e
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self,e_char, e_word, k=5):
        """ Convolutional Neural net that takes chars and spits a convoluted layer.
        Works on a batch of words
        """
        super(CNN, self).__init__()
        # conv has W (f,echar,k) & bias b
        self.conv1d = nn.Conv1d(in_channels=e_char, # embedding size of chars
                                out_channels=e_word, # output channels produced b conv (# of filters) e_word = f
                                kernel_size=k, # also called window size
                                bias=True) # size of our patch (*e_char implicit)


    def forward(self, x_reshaped: torch.Tensor) -> torch.Tensor:
        """
        @param x_reshaped: (tensor) -  size (e_char, m_word) char embedding / word size
        @returns: x_conv_out: (tensor) - shape (e_word, batch_size) """
        x_conv = self.conv1d(x_reshaped)
        x_activation = torch.relu(x_conv)
        x_conv_out, x_conv_idx = x_activation.max(dim=2)

        return x_conv_out


if __name__ == '__main__':
    batch_size = 4 # Batches of Words
    e_char = 20
    e_word = 12
    window_size = 5

    cnn = CNN(e_char,e_word,window_size)

    x_reshaped = torch.randn(batch_size,e_char,e_word)
    x_conv_out = cnn(x_reshaped).permute(1,0)
    print(x_conv_out.shape)


### END YOUR CODE

