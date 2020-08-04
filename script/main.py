from __future__ import division, print_function, absolute_import
import os

from text_cnn.train_cnn import train_cnn
from text_cnn.test_cnn import test
from misc.data_prep import *
from hyperparam import Hyperparam as hp
from transformer.train_transformer import train_transformer
from transformer.eval import eval
from seq2seq.train_model import train_seq2seq
from seq2seq.eval_model import eval_seq2seq
from non_cnn.train_model_v2 import train_seq2seq_v2
from non_cnn.eval_model_v2 import eval_seq2seq_v2

make_Data = False
train = True
inference = False

# create necessary data and save
if make_Data:
    if os.path.exists(hp.datapath):
        import shutil
        shutil.rmtree(hp.datapath)
    create_data()

# train CNN and Transformer network
if train:
    if not os.path.exists(os.path.join(hp.datapath, "rev_dict.pkl")):
        if os.path.exists(hp.datapath):
            import shutil
            shutil.rmtree(hp.datapath)
        create_data()

    train_cnn()
    #train_transformer()
    #train_seq2seq()


# get test output
if inference:
    #test()
    eval()
    #eval_seq2seq()
    #test()
