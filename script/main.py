from __future__ import division, print_function, absolute_import
import os

from text_cnn.train_cnn import train_cnn
from text_cnn.test_cnn import test
from misc.data_prep import *
from hyperparam import Hyperparam as hp
from transformer.train_transformer import train_transformer
from transformer.eval_tf import eval_tf
from seq2seq.train_model import train_seq2seq
from seq2seq.eval_model import eval_seq2seq
from non_cnn.train_model_v2 import train_seq2seq_v2
from non_cnn.eval_model_v2 import eval_seq2seq_v2


# create necessary data and save
if hp.make_data:
    if os.path.exists(hp.datapath):
        import shutil
        shutil.rmtree(hp.datapath)
    create_data()

# train CNN and Transformer network
if hp.train != "infer":

    # if data file is not made correctly, delete and renew data files
    if not os.path.exists(os.path.join(hp.datapath, "rev_dict.pkl")):
        if os.path.exists(hp.datapath):
            import shutil
            shutil.rmtree(hp.datapath)
        create_data()

    if hp.train != "non_cnn":
        train_cnn()

    if hp.train == "seq2seq":
        train_seq2seq()

    elif hp.train == "transformer":
        train_transformer()

    elif hp.train == "non_cnn":
        train_seq2seq_v2()


# get test output
if hp.train == "infer":

    if hp.infer == "cnn":
        test()
    elif hp.infer == "transformer":
        eval_tf()
    elif hp.infer == "seq2seq":
        eval_seq2seq(mode="txt")
    elif hp.infer == "non_cnn":
        eval_seq2seq_v2()
