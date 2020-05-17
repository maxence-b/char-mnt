#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch


# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(f)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output (e_word)
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1f
        e_char = 50
        window_size = 5
        dropout = 0.3
        pad_token_idx = vocab.char2id['<pad>']
        self.char_embeddings = nn.Embedding(len(vocab.char2id), e_char, pad_token_idx)
        self.cnn = CNN(e_char, embed_size, window_size)
        self.highway = Highway(embed_size, dropout)
        self.embed_size = embed_size

        ### END YOUR CODE

    def forward(self, input_tensor):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input_tensor: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1f
        x_embed = self.char_embeddings(input_tensor) #converts input into embedding (adds 4th dim for char embed size)

        max_sent, batch_size, max_word, char_embed_size = x_embed.shape #8, 4, 21, 50
        x_embed_conv = x_embed.view(max_sent * batch_size, max_word, char_embed_size) # append sentences?
        x_embed_conv = x_embed_conv.transpose(1, 2)


        x_conv_out = self.cnn(x_embed_conv)
        x_word_embed = self.highway(x_conv_out)
        x_word_embed = x_word_embed.view(max_sent, batch_size, -1)

        return x_word_embed

        ### END YOUR CODE

if __name__ == '__main__':
    from vocab import VocabEntry
    vocab = VocabEntry()

    e_word = 15
    embedding = ModelEmbeddings(e_word, vocab)
    sentence_len = 8
    batch_size = 4
    max_word_length = 21
    input_tensor = torch.randint(1,50,(sentence_len, batch_size, max_word_length)) # INTEGERS
    # print(input_tensor)
    words_embedding = embedding.forward(input_tensor)
    print(words_embedding.shape) # 8,4,15


