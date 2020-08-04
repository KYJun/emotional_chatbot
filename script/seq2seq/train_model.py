from __future__ import division, print_function, absolute_import
import os, codecs, re, sys
import tensorflow as tf
from seq2seq.seq_module import Seq2seq

sys.path.append('../')
from text_cnn.train_cnn import Cnn_Graph
from misc.data_prep import *
from hyperparam import Hyperparam as hp

def train_seq2seq():

    # load necessary data and param
    enc_input, dec_input = load_train_data("transformer", hp.datapath)
    word_dict = load_pickle(os.path.join(hp.datapath, "dict.pkl"))
    vocab_size = len(word_dict)
    maxlen = enc_input.shape[1]
    num_batch = enc_input.shape[0] // hp.batch_size

    # load cnn graph
    cnn = Cnn_Graph(vocab_size, maxlen, is_training=False)
    x_all, _ = load_train_data("transformer", hp.datapath)

    # get predicted tag with saved cnn checkpoint
    with cnn.graph.as_default():

        sess = tf.Session(graph=cnn.graph)
        cnn.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir_c))
        prediction = sess.run(cnn.predictions, feed_dict={cnn.x:x_all})
        sess.close()

    # save current prediction
    prediction.resize(prediction.shape[0], 1)
    save_pickle(prediction, os.path.join(hp.datapath, "./prediction.pkl"))

    Model = Seq2seq(vocab_size, maxlen)
    Model.train()