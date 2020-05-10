import json
import pickle
import sys

import numpy as np
import torch
import torch.nn.utils
from docopt import docopt

from char_decoder import CharDecoder
from nmt_model import NMT
from utils import pad_sents_char
from vocab import Vocab, VocabEntry



def test1():
    vocab = VocabEntry()
    print('vocab', vocab)
    sents = [['hello', 'how', 'are', 'you'], ['great', 'thanks']]
    word_ids = vocab.words2charindices(sents)
    print('Sentences in Chars', word_ids)

def test2():
    vocab = VocabEntry()
    sentences = [['Human:', 'What', 'do', 'we', 'want?'], ['Computer:', 'Natural', 'language', 'processing!'],
                 ['Human:', 'When', 'do', 'we', 'want', 'it?'], ['Computer:', 'When', 'do', 'we', 'want', 'what?']]
    word_ids = vocab.words2charindices(sentences)
    a= pad_sents_char(word_ids, 0)
    print(a)



if __name__ == '__main__':
    test2()
