from __future__ import division, print_function, absolute_import
from six.moves import xrange as range
import os, codecs
import tensorflow as tf
import numpy as np

class Hyperparam():

    ## directory ##
    make_data = True
    train = "infer" # "transformer" / "non_cnn" / "infer"
    infer = "seq2seq"

    # raw data
    textpath = "../data/emotion_train.txt"

    # processed vocab & train data
    datapath = "../data/processed_emo"

    # log dir for cnn & transformer (do not use numbers for the name)
    logdir_c = '../logdir_c'  # cnn
    logdir_t = '../logdir_t' # transformer
    logdir_s = '../logdir_s_new'

    # for inference
    fout = "../out.txt"        # out file
    ftest = "../data/test_talk.txt" # test text

    morfessor = False
    postprocess = False

    fend = "../data/end_word.txt"

    ## data description ##

    maxlen = None   # max length for one sentence; if None, maximum length from train data
    vocab_size = None   # max lexicon size; if None, all unique words are included
    num_tags = 5 # number of tag for sentence classification

    ## training ##

    batch_size = 8
    lr = 1e-3     # leanring rate
    emb_size_cnn = 256
    emb_size_sent = 248
    emb_size_tag = 8

    # for cnn
    filter_sizes = [3, 4, 5]
    num_filters = 256

    ## hidden_units = emb_size_sent + emb_size_tag
    ## for transformer
    hidden_units = 256
    num_blocks = 6  # number of encoder/decoder blocks (multi-attention)
    num_heads = 8
    dropout_prob = 0.2
    sinusoid = False

    # for sequence-to-sequence
    rnn_size = 256
    num_layers = 4
    clip = 5
    beam_width = 3

    # epochs
    num_epochs_cnn = 5
    num_epochs_main = 20
