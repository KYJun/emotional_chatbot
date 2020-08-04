from __future__ import division, print_function, absolute_import
import os, codecs, re, sys
import tensorflow as tf
from non_cnn.seq_module_v2 import Seq2seq

sys.path.append('../')
from text_cnn.train_cnn import Cnn_Graph
from misc.data_prep import *
from hyperparam import Hyperparam as hp

def train_seq2seq_v2():

    # load necessary data and param
    enc_input, dec_input = load_train_data("transformer", hp.datapath)
    word_dict = load_pickle(os.path.join(hp.datapath, "dict.pkl"))
    vocab_size = len(word_dict)
    maxlen = enc_input.shape[1]

    Model = Seq2seq(vocab_size, maxlen)
    Model.train()