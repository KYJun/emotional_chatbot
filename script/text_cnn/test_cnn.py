import tensorflow as tf
import numpy as np
import os, sys, re, codecs

sys.path.append('../')
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
                single_s = re.sub("[^가-힣.?!_ ]", "", single_s)
                single_s = re.sub(r"(\w)([.?!])", r"\1 \2", single_s)
                single_s = re.sub("[.?!]", "", single_s)
                single_s = single_s.split()
                single_s = get_token_id(single_s, vocab, add_ss="end")
                single_s = np.lib.pad(single_s, [0, maxlen-len(single_s)], 'constant', constant_values=(1, 1))
                result.append(single_s)

    return np.array(result)

def idx_to_word(idx):
    '''convert idx to word'''
    rev_dict = load_pickle(os.path.join(hp.datapath, "rev_dict.pkl"))
    result = []
    for i in idx:
        if i > 3:
            result.append(rev_dict[i])

    return " ".join(result)


def test(mode="realtime"):

    # get necessary data and param
    word_dict = load_pickle(os.path.join(hp.datapath, "dict.pkl"))
    maxlen = get_maxlen()

    # define graph
    cnn = Cnn_Graph(len(word_dict), maxlen, False)

    with tf.Session(graph=cnn.graph) as sess:

        # load from checkpoint
        cnn.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir_c))
        print('model restored')

        if mode == "realtime":
            while True:
                input_text = input('AI에게 말을 걸어 보세요 (끝내고 싶으면 X 입력해주세요) :')

                if input_text == 'X':
                    break

                else:
                    # refine input
                    input_text = re.sub("(\S)([.?!])", "\1 \2", input_text)
                    input_text = get_token_id(input_text.split(), word_dict)
                    x = np.ones(shape=[1, maxlen], dtype=np.int32)
                    x[:, :len(input_text)] = input_text

                    # predict tag
                    curr_pred = sess.run(cnn.predictions, feed_dict={cnn.x:x})
                    prediction = ["분노", "행복", "슬픔", "걱정", "심심"]
                    print(prediction[int(curr_pred)])

        else:
            test_x = refine_txt(hp.ftest, word_dict, maxlen)
            label = load_pickle(os.path.join(hp.datapath, "tag.pkl"))
            #print(label)
            label_onehot = np.eye(5)[label]
            curr_pred, acc = sess.run([cnn.predictions, cnn.accuracy], feed_dict={cnn.x:test_x, cnn.y:label_onehot})
            prediction = ["분노", "행복", "슬픔", "걱정", "심심"]
            sess.close()

            with open("cnn_out_2.txt", 'w') as f:
                for i, single_x in enumerate(test_x):
                    curr_sent = idx_to_word(single_x)
                    f.write("Sent : {}\nTag: {}\n".format(curr_sent, prediction[curr_pred[i]]))

            print("Accuracy : ", acc)


if __name__ == '__main__':
    test()