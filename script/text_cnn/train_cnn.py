import tensorflow as tf
import numpy as np
import os, sys, re

sys.path.append('../')
from misc.data_prep import *
from hyperparam import Hyperparam as hp

class Cnn_Graph():
    '''graph for text cnn'''
    def __init__(self, vocab_size, maxlen, is_training=True):

        self.graph = tf.Graph()

        with self.graph.as_default():

            # if training, directly feed tf data
            if is_training:
                self.iter = batch_tf_data("cnn", hp.datapath)
                self.x, self.y = self.iter.get_next()

            # if inference, feed data with feed dict
            else:
                self.x = tf.placeholder(tf.int32, [None, maxlen], name='input')
                self.y = tf.placeholder(tf.float32, [None, hp.num_tags], name='label')

            x = self.x

            # embedding for words
            with tf.name_scope('embedding'):
                self.Emb = tf.Variable(tf.random_uniform([vocab_size, hp.emb_size_cnn], -1.0, 1.0), name='emb')
                # [None, sequence_length, embedding_size]
                embedded_chars = tf.nn.embedding_lookup(self.Emb, x)
                # [None, sequence_length, embedding_size, 1]
                embedded_chars = tf.expand_dims(embedded_chars, -1)
                print(embedded_chars)

            # convolution
            pooled_outputs = []

            for i, filter_size in enumerate(hp.filter_sizes):
                with tf.name_scope('conv-maxpool-%s' % filter_size):
                    # convolution
                    filter_shape = [filter_size, hp.emb_size_cnn, 1, hp.num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                    b = tf.Variable(tf.constant(0.1, shape=[hp.num_filters]), name='b')
                    conv = tf.nn.conv2d(
                        embedded_chars,
                        W,
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name='conv')
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                    #print(h)
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, maxlen - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name='pool')
                    pooled_outputs.append(pooled)
                    #print(pooled)

            # concat output from 3 filters
            num_filters_total = hp.num_filters * len(hp.filter_sizes)
            h_pool = tf.concat(pooled_outputs, 3)
            h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

            # dropout
            with tf.name_scope('dropout'):
                h_drop = tf.layers.dropout(h_pool_flat, hp.dropout_prob, training=tf.convert_to_tensor(is_training))

            # prediction
            with tf.name_scope('output'):
                W = tf.get_variable('W', shape=[num_filters_total, hp.num_tags],
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[hp.num_tags]), name='b')

                scores = tf.nn.xw_plus_b(h_drop, W, b, name='scores')
                self.predictions = tf.argmax(scores, 1, name='predictions')

            # gain accuracy
            with tf.name_scope('accuracy'):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')

            # calculate loss and minimize by adam optimizer
            if is_training:
                with tf.name_scope('loss'):
                    losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=scores)
                    self.loss = tf.reduce_mean(losses)
                    self.optimizer = tf.train.AdamOptimizer(hp.lr).minimize(self.loss)

            #define saver
            self.saver = tf.train.Saver()

def train_cnn():

    # load necessary data and parameter
    input_data, target_data = load_train_data("cnn", hp.datapath)
    word_dict = load_pickle(os.path.join(hp.datapath, "dict.pkl"))
    vocab_size = len(word_dict)
    maxlen = input_data.shape[1]
    num_batch = input_data.shape[0] // hp.batch_size

    # define graph
    cnn = Cnn_Graph(vocab_size, maxlen)
    print("Graph loaded")

    with tf.Session(graph=cnn.graph) as sess:

        # load from checkpoint
        if tf.train.get_checkpoint_state(hp.logdir_c):
            cnn.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir_c))
            start_epoch, global_step = re.findall("[0-9]+", tf.train.latest_checkpoint(hp.logdir_c))
            start_epoch = int(start_epoch)+1
            global_step = int(global_step)
            #print(start_epoch, global_step)
            print('model restored')

        # or initialize
        else:
            sess.run(tf.global_variables_initializer())
            global_step = 0
            start_epoch = 1

        for epoch in range(start_epoch, hp.num_epochs_cnn+1):

            epoch_loss = 0

            # initialize batch iteratior
            sess.run(cnn.iter.initializer)

            for step in range(num_batch):

                # optimize
                _, step_loss = sess.run([cnn.optimizer, cnn.loss])
                epoch_loss += step_loss
                global_step += 1

            print("Current Epoch : {:02d} Loss: {:.4f}".format(epoch, epoch_loss/num_batch))

            if epoch % 5 == 0:
                cnn.saver.save(sess, hp.logdir_c + '/model_epoch_{:02d}_gs_{:d}' .format(epoch, global_step))

if __name__ == '__main__':
    os.chdir("../")
    train_cnn()

