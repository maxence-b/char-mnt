#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1d
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F


class Highway(nn.Module):

    # TODO: support for batches
    def __init__(self, e_word, dropout_rate=0.3):
        '''from x_conv_out
         to x_highway
         uses bathces'''
        super(Highway, self).__init__()
        self.dropout_rate = dropout_rate
        self.w_proj = nn.Linear(e_word, e_word, bias=True)
        self.w_gate = nn.Linear(e_word, e_word, bias=True)
        self.dropout = nn.Dropout(self.dropout_rate)


    def forward(self, x_conv_out: torch.Tensor) -> torch.Tensor:
        '''input: x_conv_out
        outputs: x_word_embedding'''
        x_proj = F.relu(self.w_proj(x_conv_out))
        x_gate = F.sigmoid(self.w_gate(x_conv_out))

        x_highway = x_gate * x_proj + (1 - x_gate) * x_conv_out
        x_word_emb = self.dropout(x_highway)




### END YOUR CODE

