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
    sents = [['hello','how','are','you'], ['great','thanks']]


if __name__ == '__main__':
    test1()
