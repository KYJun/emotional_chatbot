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

def refine_txt(source, vocab, maxlen):
    '''refine test file for input (padded index array)'''
    result = []
    with codecs.open(source, "r", encoding='utf-8') as f:
        s = f.readlines()
        for single_s in s:
            if len(single_s.split()) <= maxlen:
                single_s = re.sub("[^가-힣.?! ]", "", single_s)
                single_s = re.sub(r"(\w)([.?!])", r"\1 \2", single_s)
                single_s = re.sub("[.?!]", "", single_s)
                single_s = single_s.split()
                single_s = get_token_id(single_s, vocab, add_ss=True)
                single_s = np.lib.pad(single_s, [0, maxlen-len(single_s)], 'constant', constant_values=(0, 0))
                result.append(single_s)
    return np.array(result)

def generate_answer(source, vocab_size, maxlen):
    '''generate answer for given test file'''

    # predict tag for given test sentence
    cnn = Cnn_Graph(vocab_size, maxlen, is_training=False)
    print("cnn graph loading...")
    with cnn.graph.as_default():
        sess = tf.Session(graph=cnn.graph)
        cnn.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir_c))
        tag_pred = sess.run(cnn.predictions, feed_dict={cnn.x: source})
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
        feed_dict = {transformer.x: source, transformer.tag: tag_pred, transformer.y: pseudo_targets}
        answer_pred = sess.run(transformer.preds, feed_dict=feed_dict)
        sess.close()

    print("Answer predicted")

    return tag_pred, answer_pred


def idx_to_word(idx):
    '''convert idx to word'''
    rev_dict = load_pickle(os.path.join(hp.datapath, "rev_dict.pkl"))
    result = []
    for i in idx:
        if i > 3:
            result.append(rev_dict[i])

    return " ".join(result)

def save_result(fout, source, tag, answer):
    '''save result (original sent, predicted tag, predicted answer)'''

    with codecs.open(fout, 'w', encoding='utf-8') as f:
        for single_s, single_t, single_a in zip(source, tag, answer):
            f.write("Question : "+ idx_to_word(single_s)+"\n")
            f.write("Tag : " + str(single_t) +"\n")
            if hp.postprocess:
                single_new_a = process_answer(single_a)
                single_a = process_end(idx_to_word(single_new_a))
            single_a = idx_to_word(single_a)
            f.write("Answer : " + single_a +"\n")

def process_answer(answer):

    new_a = []
    for single_a in answer:
        if single_a > 3:
            if not new_a:
                new_a.append(single_a)
            elif single_a != new_a[-1]:
                new_a.append(single_a)
    return new_a

def process_end(answer):

    answer_split = answer.split()

    with open(hp.fend) as f:
        endwords = f.readlines()
        end_idx = []
        endwords = [endword.strip() for endword in endwords]

        for idx, word in enumerate(answer_split):
            if word in endwords:
                end_idx.append(idx)

        if end_idx:
            end_idx = min(end_idx)
            new_a = " ".join(answer_split[:end_idx+1])

        else:
            new_a = answer

    return new_a


def eval_tf():
    # get necessary data and param
    maxlen = get_maxlen()
    word_dict = load_pickle(os.path.join(hp.datapath, "dict.pkl"))
    vocab_size = len(word_dict)

    # process -> get tag -> get answer -> save as txt file
    source = refine_txt(hp.ftest, word_dict, maxlen)
    pred_tag, pred_answer = generate_answer(source, vocab_size, maxlen)
    save_result(hp.fout, source, pred_tag, pred_answer)


if __name__ == '__main__':
    os.chdir("../")
    eval_tf()
    print("Done")


