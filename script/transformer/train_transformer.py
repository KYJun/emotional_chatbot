from __future__ import division, print_function, absolute_import
from six.moves import xrange as range
import os, codecs, re, sys
import tensorflow as tf
import numpy as np
from transformer.transformer_module import *

sys.path.append('../')
from text_cnn.train_cnn import Cnn_Graph
from misc.data_prep import *
from hyperparam import Hyperparam as hp



class Tr_Graph():
    '''define transformer graph'''
    def __init__(self, is_training, vocab_size, maxlen):

        self.graph = tf.Graph()

        with self.graph.as_default():

            # if training, use tf dataset as data feeder
            if is_training:
                self.iter = batch_tf_data("transformer", hp.datapath)
                self.x, self.y, self.tag = self.iter.get_next()

            # if inference, use placeholder
            else:
                self.x = tf.placeholder(tf.int32, [None, maxlen], name='enc_input')
                self.y = tf.placeholder(tf.int32, [None, maxlen], name='dec_input')
                self.tag = tf.placeholder(tf.int32, [None, 1], name='tag_input')

            x = self.x[:, 1:]
            tag = tf.tile(self.tag, multiples=[1, maxlen])
            #print(self.tag)

            # define decoder inputs
            self.decoder_inputs = self.y


            with tf.variable_scope("embedding"):
                word_table = tf.get_variable('lookup_table',
                                             dtype=tf.float32,
                                             shape=[vocab_size, hp.emb_size_sent],
                                             initializer=tf.contrib.layers.xavier_initializer())
                tag_table = tf.get_variable('tag_table',
                                            dtype=tf.float32,
                                            shape=[vocab_size, hp.emb_size_tag],
                                            initializer=tf.contrib.layers.xavier_initializer())
                self.enc = tf.nn.embedding_lookup(word_table, x)
                self.dec = tf.nn.embedding_lookup(word_table, self.decoder_inputs)
                self.tag_emb = tf.nn.embedding_lookup(tag_table, tag)

            # Encoder
            with tf.variable_scope("encoder"):

                ## Positional Encoding
                if hp.sinusoid:
                    self.enc += positional_encoding(x,
                                                    num_units=hp.emb_size_sent,
                                                    zero_pad=False,
                                                    scale=False,
                                                    scope="enc_pe")
                else:
                    self.enc += embedding(
                        tf.tile(tf.expand_dims(tf.range(tf.shape(x)[1]), 0), [tf.shape(x)[0], 1]),
                        vocab_size=maxlen,
                        num_units=hp.emb_size_sent,
                        zero_pad=False,
                        scale=False,
                        scope="enc_pe")

                # concat both embedding vector into one tensor
                self.total_enc = tf.concat((self.enc, self.tag_emb[:,1:,:]), axis=-1)
                self.enc = self.total_enc
                #print(self.enc)

                ## Dropout
                self.enc = tf.layers.dropout(self.enc,
                                             rate=hp.dropout_prob,
                                             training=tf.convert_to_tensor(is_training))

                ## Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        ### Multihead Attention
                        print(self.enc)
                        self.enc = multihead_attention(queries=self.enc,
                                                       keys=self.enc,
                                                       num_units=hp.hidden_units,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=hp.dropout_prob,
                                                       is_training=is_training)

                        ### Feed Forward
                        self.enc = feedforward(self.enc, num_units=[4 * hp.hidden_units, hp.hidden_units])

            # Decoder
            with tf.variable_scope("decoder"):

                ## Positional Encoding
                if hp.sinusoid:
                    self.dec += positional_encoding(self.decoder_inputs,
                                                    vocab_size=maxlen,
                                                    num_units=hp.emb_size_sent,
                                                    zero_pad=False,
                                                    scale=False,
                                                    scope="dec_pe")
                else:
                    self.dec += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.decoder_inputs)[1]), 0),
                                                  [tf.shape(self.decoder_inputs)[0], 1]),
                                          vocab_size=maxlen,
                                          num_units=hp.emb_size_sent,
                                          zero_pad=False,
                                          scale=False,
                                          scope="dec_pe")

                self.total_dec = tf.concat((self.dec, self.tag_emb), axis=-1)
                self.dec = self.total_dec

                ## Dropout
                self.dec = tf.layers.dropout(self.dec,
                                             rate=hp.dropout_prob,
                                             training=tf.convert_to_tensor(is_training))

                ## Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        ## Multihead Attention ( self-attention)
                        self.dec = multihead_attention(queries=self.dec,
                                                       keys=self.dec,
                                                       num_units=hp.hidden_units,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=hp.dropout_prob,
                                                       is_training=is_training,
                                                       scope="self_attention")

                        ## Multihead Attention ( vanilla attention)
                        self.dec = multihead_attention(queries=self.dec,
                                                       keys=self.enc,
                                                       num_units=hp.hidden_units,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=hp.dropout_prob,
                                                       is_training=is_training,
                                                       scope="vanilla_attention")

                        ## Feed Forward
                        self.dec = feedforward(self.dec, num_units=[4 * hp.hidden_units, hp.hidden_units])

            # Final linear projection
            self.logits = tf.layers.dense(self.dec, vocab_size)
            self.probs = tf.nn.softmax(self.logits)
            self.preds = tf.to_int32(tf.argmax(self.logits, axis=-1))
            self.istarget = tf.to_float(tf.not_equal(self.y, 0))
            self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y)) * self.istarget) / (
                tf.reduce_sum(self.istarget))

            if is_training:
                # Loss
                self.y_smoothed = label_smoothing(tf.one_hot(self.y, depth=vocab_size))
                self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y_smoothed)
                self.mean_loss = tf.reduce_sum(self.loss * self.istarget) / (tf.reduce_sum(self.istarget))

                # Training Scheme
                self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
                self.train_op = self.optimizer.minimize(self.mean_loss)

            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()


def train_transformer():

    # load necessary data and param
    enc_input, dec_input = load_train_data("transformer", hp.datapath)
    word_dict = load_pickle(os.path.join(hp.datapath, "dict.pkl"))
    vocab_size = len(word_dict)
    maxlen = enc_input.shape[1]
    num_batch = enc_input.shape[0] // hp.batch_size

    # load cnn graph
    cnn = Cnn_Graph(vocab_size, maxlen, is_training=False)
    x_all, y_all = load_train_data("transformer", hp.datapath)

    # get predicted tag with saved cnn checkpoint
    with cnn.graph.as_default():

        sess = tf.Session(graph=cnn.graph)
        cnn.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir_c))
        prediction = sess.run(cnn.predictions, feed_dict={cnn.x:x_all})
        sess.close()

    # save current prediction
    prediction.resize(prediction.shape[0], 1)
    save_pickle(prediction, os.path.join(hp.datapath, "./prediction.pkl"))

    # load transformer graph
    transformer = Tr_Graph(is_training=True, vocab_size=vocab_size, maxlen=maxlen)
    print("Transformer Graph loaded")

    with transformer.graph.as_default():

        with tf.Session(graph=transformer.graph) as sess:

            # load from checkpoint
            if tf.train.get_checkpoint_state(hp.logdir_t):
                transformer.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir_t))
                start_epoch, global_step = re.findall("[0-9]+", tf.train.latest_checkpoint(hp.logdir_t))
                start_epoch = int(start_epoch)+1
                global_step = int(global_step)
                print('Model restored')

                if hp.num_epochs_trans > start_epoch:
                    print("Starting from epoch : ", start_epoch)

            # or initialize
            else:
                sess.run(transformer.init)
                global_step = 0
                start_epoch = 1
                print("Starting from scratch")

            for epoch in range(start_epoch, hp.num_epochs_trans+1):

                epoch_loss = 0
                # initialize iterator
                sess.run(transformer.iter.initializer)

                for step in range(num_batch):

                    # optimize
                    _, step_loss = sess.run([transformer.train_op, transformer.mean_loss])
                    epoch_loss += step_loss
                    global_step += 1

                print("Current Epoch : {:02d} Loss: {:.4f}".format(epoch, epoch_loss/num_batch))

                if epoch % 5 == 0:
                    transformer.saver.save(sess, hp.logdir_t + '/model_epoch_{:02d}_gs_{:d}' .format(epoch, global_step))

            print("Training Done!")

if __name__ == "__main__":
    os.chdir("../")
    train_transformer()