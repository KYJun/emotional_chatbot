from __future__ import division, print_function, absolute_import
from six.moves import xrange as range
import os, sys, re, codecs
import tensorflow as tf
import numpy as np
from transformer.train_transformer import Tr_Graph

sys.path.append("../")
from text_cnn.train_cnn import Cnn_Graph
from misc.data_prep import *
from hyperparam import Hyperparam as hp

def test():
    maxlen = get_maxlen()
    word_dict = load_pickle(os.path.join(hp.datapath, "dict.pkl"))
    vocab_size = len(word_dict)

    single_s = "상장 받았어"
    single_s = re.sub("[^가-힣.?! ]", "", single_s)
    single_s = re.sub(r"(\w)([.?!])", r"\1 \2", single_s)
    single_s = re.sub("[.?!]", "", single_s)
    single_s = single_s.split()
    single_s = get_token_id(single_s, word_dict, add_ss=True)
    single_s = np.lib.pad(single_s, [0, maxlen-len(single_s)], 'constant', constant_values=(0, 0))
    single_s.resize([1, 7])

    # predict tag for given test sentence
    cnn = Cnn_Graph(vocab_size, maxlen, is_training=False)
    print("cnn graph loading...")
    with cnn.graph.as_default():
        sess = tf.Session(graph=cnn.graph)
        cnn.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir_c))
        tag_pred = sess.run(cnn.predictions, feed_dict={cnn.x: single_s})
        sess.close()

    print("tag predicted")
    tag_pred.resize(tag_pred.shape[0], 1)

    # generate answer with given test sentence and predicted tags
    transformer = Tr_Graph(is_training=False, vocab_size=vocab_size, maxlen=maxlen)
    print("Transformer Graph loaded")

    pseudo_targets = np.zeros(shape=(tag_pred.shape[0], maxlen), dtype=np.int32)

    with transformer.graph.as_default():
        sess = tf.Session(graph=transformer.graph)
        transformer.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir_t))
        feed_dict = {transformer.x: single_s, transformer.tag: tag_pred, transformer.y: pseudo_targets}
        answer_pred, probs_pred = sess.run([transformer.preds, transformer.probs], feed_dict=feed_dict)
        sess.close()

    print("Answer predicted")
    print(idx_to_word(answer_pred[0]))

    sorted_idx = np.argsort(probs_pred[0], axis=-1)[:, -3:]

    with open("output.txt", "w") as f:
        for single_row in probs_pred[0]:
            for single_num in single_row:
                f.write(str(single_num))
                f.write("  ")
            f.write("\n")

    with open("out_top3.txt", "w") as f:
        rev_dict = load_pickle(os.path.join(hp.datapath, "rev_dict.pkl"))
        for single_row in sorted_idx:
            for single_num in single_row:
                f.write(str(single_num)+" : "+rev_dict[single_num] + " , ")
            f.write("\n")

    with open("beam_search.txt", "w") as f:
        result = []
        for single_row in sorted_idx:
            rand_idx = np.random.randn(3)
            result.append(single_row[rand_idx])
        result = idx_to_word(result)
        f.write(result)



def idx_to_word(idx):
    '''convert idx to word'''
    rev_dict = load_pickle(os.path.join(hp.datapath, "rev_dict.pkl"))
    result = []
    for i in idx:
        if i > 3:
            result.append(rev_dict[i])

    return " ".join(result)