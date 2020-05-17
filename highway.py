#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1d
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F

class Highway(nn.Module):

    def __init__(self, e_word, dropout_rate=0.3):
        """Highway network from convoluted char output to word embedding
        @param e_word: int:  size of word embedding
        @param dropout_rate: (float): Dropout probability"""
        super(Highway, self).__init__()

        self.dropout_rate = dropout_rate
        self.w_proj = nn.Linear(e_word, e_word, bias=True)
        self.w_gate = nn.Linear(e_word, e_word, bias=True)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x_conv_out: torch.Tensor) -> torch.Tensor:
        """
        @param: x_conv_out: (tensor) - shape (e_word,batch_size)
        @returns: x_word_emb: (Tensor) tensor) - shape (e_word) the output word embedding """

        x_proj = F.relu(self.w_proj(x_conv_out))
        x_gate = F.sigmoid(self.w_gate(x_conv_out))

        x_highway = x_gate * x_proj + (1 - x_gate) * x_conv_out
        x_word_emb = self.dropout(x_highway)

        return x_word_emb


if __name__ == '__main__':
    e_word = 12 # e_word = 256 but let's see with a baby 12
    batch_size = 4
    highway = Highway(e_word)
    x_conv_out = torch.randn(e_word, batch_size)# embedding per batch of 4 words
    print(x_conv_out.size())
### END YOUR CODE

