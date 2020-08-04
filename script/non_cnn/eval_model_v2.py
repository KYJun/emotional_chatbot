from __future__ import division, print_function, absolute_import
from six.moves import xrange as range
import os, sys, re, codecs
import tensorflow as tf
import numpy as np
from non_cnn.seq_module_v2 import Seq2seq

sys.path.append("../")
from misc.data_prep import *
from hyperparam import Hyperparam as hp

def refine_txt(source, vocab, maxlen):
    '''refine test file for input (padded index array)'''
    result = []
    s_lengths = []
    with codecs.open(source, "r", encoding='utf-8') as f:
        s = f.readlines()
        for single_s in s:
            if len(single_s.split()) <= maxlen:
                #single_s = re.sub("[^가-힣.?!_ ]", "", single_s)
                single_s = re.sub(r"(\w)([.?!])", r"\1 \2", single_s)
                single_s = re.sub("[.?!]", "", single_s)
                single_s = single_s.split()
                s_lengths.append(len(single_s)+1)
                single_s = get_token_id(single_s, vocab, add_ss="end")
                single_s = np.lib.pad(single_s, [0, maxlen-len(single_s)], 'constant', constant_values=(1, 1))
                result.append(single_s)

    return np.array(result), np.array(s_lengths)


def idx_to_word(idx):
    '''convert idx to word'''
    rev_dict = load_pickle(os.path.join(hp.datapath, "rev_dict.pkl"))
    result = []
    for i in idx:
        if i > 3:
            result.append(rev_dict[i])

    return " ".join(result)

def save_result(fout, source, answer, beam_width=0, post_process=True, morfessor=False):
    '''save result (original sent, predicted answer)'''
    tags = ["부정", "긍정"]

    with codecs.open(fout, 'w', encoding='utf-8') as f:
        for j, (single_s, single_a) in enumerate(zip(source, answer)):
            f.write("Question {} : {}\n".format(j, process_answer(single_s, morfessor)))

            if beam_width > 0:
                single_a = list(np.transpose(np.array(single_a)))
                for i, answer in enumerate(single_a):
                    answer = process_answer(answer, morfessor, post_process)
                    f.write("Answer {} : {}\n".format(i+1, answer))
            else:
                answer = process_answer(answer, morfessor, post_process)
                f.write("Answer : " + answer + "\n")
            f.write("\n")

def process_repeat(answer):

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

def process_answer(answer, morfessor=False, post_process=False):

    if post_process:
        answer = process_repeat(answer)

    answer = idx_to_word(answer)

    if morfessor:
        answer = re.sub(r"\_ \_", "", answer)
        answer = re.sub(r"\_", "", answer)

    if post_process:
        answer = process_end(answer)

    return answer


def eval_seq2seq_v2():
    # get necessary data and param
    maxlen = get_maxlen()
    word_dict = load_pickle(os.path.join(hp.datapath, "dict.pkl"))
    vocab_size = len(word_dict)

    # process -> get answer -> save as txt file
    source, s_lengths = refine_txt(hp.ftest, word_dict, maxlen)

    Model = Seq2seq(vocab_size=vocab_size, maxlen=maxlen)

    with Model.graph.as_default():
        Model.build_graph(is_training=False)

    with tf.Session(graph=Model.graph) as sess:
        Model.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir_s))
        print('Model restored')
        feed_dict = {Model.x: source, Model.x_lengths: s_lengths}
        infer_logits, sample_ids = sess.run([Model.logits, Model.sample_id], feed_dict=feed_dict)
        sess.close()

    save_result(hp.fout, source, sample_ids, hp.beam_width, hp.morfessor, hp.postprocess)
    print("File saved")