import os, codecs, re, pickle, sys
import numpy as np
import tensorflow as tf
from collections import Counter

sys.path.append('../')
from hyperparam import Hyperparam as hp

def read_raw_data(filename, tag_confirm=True):
    '''read text data'''
    
    input_data = []
    target_data = []
    tag = []
    word_list = []

    with codecs.open(filename, 'r', encoding='utf-8') as f:
        print('loading data')

        total_lines = f.readlines()
        print(len(total_lines))
        for single_line in total_lines:

            # make space between periods(.?!)
            single_line = re.sub(r"(\w)([.?!])", r"\1 \2", single_line)

            # delete none-words (temporarily)
            single_line = re.sub("[.?!]", "", single_line)
            single_line = single_line.split()

            # collect human input
            if single_line[0] == "A:":
                input_data.append(single_line[1:])
                word_list.extend(single_line[1:])

            # collect ai output and senetence tag
            elif single_line[0] == "B:":
                if tag_confirm:
                    tag.append(single_line[-1])
                    single_line = single_line[:-1]
                else:
                    tag.append(1)
                
                target_data.append(single_line[1:])
                word_list.extend(single_line[1:])

            else:
                print(single_line)

    return input_data, target_data, tag, word_list

def save_length(input_data, target_data):
    '''save length of each data into pkl data'''
    x_lengths = [len(single_i)+1 for single_i in input_data]
    y_lengths = [len(single_t)+1 for single_t in target_data]
    save_pickle(x_lengths, os.path.join(hp.datapath, "xl.pkl"))
    save_pickle(y_lengths, os.path.join(hp.datapath, "yl.pkl"))


def build_vocab(word_list):
    '''build vocabulary(lexicon) out of unique words'''

    print('building vocabulary')
    vocab = dict()

    # add necessary tokens
    vocab['_UNK'] = 0   # for oov
    vocab['_PAD'] = 1   # for sentence padding
    vocab['_STR'] = 2   # start token
    vocab['_END'] = 3   # end token

    # get vocabulary size; if none, take all the words
    if hp.vocab_size == None:
        vocab_size = len(list(set(word_list)))
    else:
        vocab_size = hp.vocab_size
    print(vocab_size)

    # get frequency of each words
    freq_word_list = [word[0] for word in Counter(word_list).most_common(vocab_size)]
    write_word_list = [word[0] + " " + str(word[1]) for word in Counter(word_list).most_common(vocab_size)]

    # build vocabulary
    for i, word in enumerate(freq_word_list):
        vocab[word] = i+4

    # build reverse dictionary for inference
    rev_vocab = ["_UNK", "_PAD", "_STR", "_END"]
    rev_vocab.extend(freq_word_list)

    with open("word_list.txt", "w") as f:
        for single_contents in write_word_list:
            f.write(single_contents+"\n")
    print(freq_word_list[:10])
    return vocab, rev_vocab

def get_token_id(tokens, vocab, add_ss=None):
    '''convert words to index token or vice versa'''

    result = []
    for key in tokens:
        try:
            value = vocab[key]
        except:
            value = 0
        result.append(value)

    # add start and end token
    if add_ss=="str":
        result.insert(0, 2)

    elif add_ss=="end":
        result.append(3)

    elif add_ss=="all":
        result.insert(0, 2)
        result.append(3)

    else:
        result = result

    return np.array(result) 


def build_input(x_data, y_data, vocab):
    '''convert each sentence into padded index array'''

    # get max length; if None, get the max length from train data
    if not hp.maxlen:
        maxlen = max(max([len(single_x) for single_x in x_data]), max([len(single_y) for single_y in y_data]))+1
    else:
        maxlen = hp.maxlen+1

    # convert each sentence to index array
    x_idx_data = [get_token_id(single_x, vocab, add_ss="end") for single_x in x_data if len(single_x) <= maxlen]
    y_idx_data = [get_token_id(single_y, vocab, add_ss="str") for single_y in y_data if len(single_y) <= maxlen]
    z_idx_data = [get_token_id(single_y, vocab, add_ss="end") for single_y in y_data if len(single_y) <= maxlen]

    # padding for each array
    X = np.zeros([len(x_idx_data), maxlen], np.int32)
    Y = np.zeros([len(y_idx_data), maxlen], np.int32)
    Z = np.zeros([len(z_idx_data), maxlen], np.int32)

    for i, (x, y, z) in enumerate(zip(x_idx_data, y_idx_data, z_idx_data)):
        X[i] = np.lib.pad(x, [0, maxlen-len(x)], 'constant', constant_values=(1, 1))
        Y[i] = np.lib.pad(y, [0, maxlen-len(y)], 'constant', constant_values=(1, 1))
        Z[i] = np.lib.pad(z, [0, maxlen-len(z)], 'constant', constant_values=(1, 1))

    return X, Y, Z


def save_pickle(data, filepath):
    '''save data into pickle'''

    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(filepath):
    '''load pickled data'''

    with open(filepath, 'rb') as f:
        file = pickle.load(f)
    return file

def create_data(unit="eojol"):
    '''main function'''

    input_data, target_data, tag, unique_words = read_raw_data(hp.textpath)

    # build vocabulary
    vocab, rev_vocab = build_vocab(unique_words)
    # make padded token index
    X, Y, Z = build_input(input_data, target_data, vocab)
    # get tags from each sentence
    tag = np.array(tag)

    # save processed data into datapath
    if not os.path.exists(hp.datapath): os.mkdir(hp.datapath)
    save_pickle(X, os.path.join(hp.datapath, "x.pkl"))
    save_pickle(Y, os.path.join(hp.datapath, "y.pkl"))
    save_pickle(Z, os.path.join(hp.datapath, "z.pkl"))
    save_pickle(tag, os.path.join(hp.datapath, "tag.pkl"))
    save_pickle(vocab, os.path.join(hp.datapath, "dict.pkl"))
    save_pickle(rev_vocab, os.path.join(hp.datapath, "rev_dict.pkl"))
    #print(X[0], Y[0], Z[0])

    # save input and target data length (for seq2seq)
    save_length(input_data, target_data)

def get_vocabsize():
    '''get vocabulary size if none'''

    vocab_size = len(load_pickle(os.path.join(os.path.join(hp.datapath, "dict.pkl"))))

    return vocab_size

def get_maxlen():
    '''get max length if none'''

    maxlen = load_pickle(os.path.join(hp.datapath, "x.pkl")).shape[1]

    return maxlen

def load_train_data(model, filepath):
    '''load data and process based on model'''

    # load human input and ai output
    x_filepath = os.path.join(filepath, "x.pkl")
    y_filepath = os.path.join(filepath, "y.pkl")
    input_data_x = load_pickle(x_filepath)
    input_data_y = load_pickle(y_filepath)

    # for cnn, concatenate both sentence data as input and tag as target
    if model == "cnn":
        #input_data = np.concatenate((input_data_x, input_data_y), axis=0)
        input_data = np.array(input_data_x)

        # load tag data and convert into one-hot vector
        tag_filepath = os.path.join(filepath, "tag.pkl")
        target_data = load_pickle(tag_filepath)
        target_data = np.array([int(i) for i in target_data])
        target_data = np.eye(hp.num_tags)[target_data]

        # double tag data
        #target_data = np.concatenate((target_data, target_data), axis=0)

    # for transformer, human input as input and ai output as target
    else:
        input_data = np.array(input_data_x)
        target_data = np.array(input_data_y)

    return input_data, target_data


def batch_tf_data(model, filepath):
    '''create tf dataset based on the model'''

    # load train data based on the model
    x, y = load_train_data(model=model, filepath=filepath)

    # if cnn, input and target needed
    if model == "cnn":
        dataset = tf.data.Dataset.from_tensor_slices((x,y))

    # if transformer, input, target and tag data needed
    else:
        # tag data is predicted from cnn and saved
        tag_pred = load_pickle(os.path.join(filepath, "prediction.pkl"))
        dataset = tf.data.Dataset.from_tensor_slices((x,y, tag_pred))

    # shuffle and get batch size
    dataset = dataset.shuffle(buffer_size=len(x))
    dataset = dataset.batch(hp.batch_size).repeat()

    # define iterator
    iter = dataset.make_initializable_iterator()

    return iter


def batch_seq2seq(filepath, tagging=True):
    '''get batch iterator for seq2seq training
    : input, target, length of each, predicted tag'''

    x_filepath = os.path.join(filepath, "x.pkl")
    y_filepath = os.path.join(filepath, "y.pkl")
    z_filepath = os.path.join(filepath, "z.pkl")
    xl_filepath = os.path.join(filepath, "xl.pkl")
    yl_filepath = os.path.join(filepath, "yl.pkl")

    x = load_pickle(x_filepath)
    y = load_pickle(y_filepath)
    z = load_pickle(z_filepath)
    xl = load_pickle(xl_filepath)
    yl = load_pickle(yl_filepath)

    if tagging:
        tag_pred = load_pickle(os.path.join(filepath, "prediction.pkl"))
        dataset = tf.data.Dataset.from_tensor_slices((x, y, z, xl, yl, tag_pred))

    else:
        dataset = tf.data.Dataset.from_tensor_slices((x, y, z, xl, yl))

    dataset = dataset.shuffle(buffer_size=len(x))
    dataset = dataset.batch(hp.batch_size).repeat()

    # define iterator
    iter = dataset.make_initializable_iterator()

    return iter


if __name__ == '__main__':
    os.chdir("../")
    if not os.path.exists(os.path.join(hp.datapath, "rev_dict.pkl")):

        if os.path.exists(hp.datapath):
            import shutil
            shutil.rmtree(hp.datapath)

        create_data()
    