from __future__ import absolute_import, division, print_function
from six.moves import xrange as range
import os, codecs, re, sys

import tensorflow as tf
from tensorflow.python.layers.core import Dense
import numpy as np

sys.path.append("../")
from misc.data_prep import *
from hyperparam import Hyperparam as hp

class Seq2seq(object):

    def __init__(self, vocab_size, maxlen):

        self.graph = tf.Graph()
        self.vocab_size = vocab_size
        self.maxlen = maxlen

    def build_graph(self, is_training):

        if is_training:
            keep_prob = 1 - hp.dropout_prob
        else:
            keep_prob = 1

        if is_training:
            self.iter = batch_seq2seq(hp.datapath, tagging=False)
            self.x, self.y, self.z, self.x_lengths, self.y_lengths = self.iter.get_next()

        else:
            self.x = tf.placeholder(tf.int32, [None, self.maxlen], name='enc_input')
            self.x_lengths = tf.placeholder(tf.int32, [None], name='enc_length')
            self.y = tf.placeholder(tf.int32, [None, self.maxlen], name='dec_input')
            self.z = tf.placeholder(tf.int32, [None, self.maxlen], name='dec_target')
            self.y_lengths = tf.placeholder(tf.int32, [None], name='dec_length')

        vocab_size = self.vocab_size
        maxlen = self.maxlen

        enc_inputs = self.x
        dec_inputs = self.y
        dec_targets = self.z


        with tf.variable_scope("embedding"):
            word_emb = tf.get_variable('lookup_table',
                                           dtype=tf.float32,
                                           shape=[vocab_size, hp.emb_size_sent],
                                           initializer=tf.contrib.layers.xavier_initializer())

            #if zero_pad:
            #    word_emb = tf.concat((tf.zeros(shape=[1, hp.emb_size_sent]),
            #                              lookup_table[1:, :]), 0)

            self.enc = tf.nn.embedding_lookup(word_emb, enc_inputs)

            if is_training:
                self.dec = tf.nn.embedding_lookup(word_emb, dec_inputs)


        with tf.variable_scope("encoder"):
            ## Embedding
            self.total_enc = self.enc
            #self.total_enc = self.enc

            encoder_cell_fw = tf.nn.rnn_cell.MultiRNNCell([self.make_rnn_cell(hp.rnn_size//2, keep_prob) for _ in range(hp.num_layers)])
            encoder_cell_bw = tf.nn.rnn_cell.MultiRNNCell([self.make_rnn_cell(hp.rnn_size//2, keep_prob) for _ in range(hp.num_layers)])

            outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell_fw,
                cell_bw=encoder_cell_bw, inputs=self.total_enc, sequence_length=self.x_lengths, dtype=tf.float32)

            out_fw, out_bw = outputs
            state_fw, state_bw = states

            self.encoder_outputs = tf.concat((out_fw, out_bw), -1)
            encoder_state = []

            for i in range(hp.num_layers):
                bi_state_c = tf.concat((state_fw[i].c, state_bw[i].c), -1)
                bi_state_h = tf.concat((state_fw[i].h, state_bw[i].h), -1)
                bi_lstm_state = tf.nn.rnn_cell.LSTMStateTuple(c=bi_state_c, h=bi_state_h)
                encoder_state.append(bi_lstm_state)

            self.encoder_state = tuple(encoder_state)

        with tf.variable_scope("Decoder"):

            batch_size = tf.shape(enc_inputs)[0]
            seq_lengths = self.x_lengths

            if not is_training and hp.beam_width > 0:
                self.encoder_outputs = tf.contrib.seq2seq.tile_batch(self.encoder_outputs, multiplier=hp.beam_width)
                self.encoder_state = tf.contrib.seq2seq.tile_batch(self.encoder_state, multiplier=hp.beam_width)
                seq_lengths = tf.contrib.seq2seq.tile_batch(self.x_lengths, multiplier=hp.beam_width)
                batch_size *= hp.beam_width


            decoder_cell = tf.nn.rnn_cell.MultiRNNCell([self.make_rnn_cell(hp.rnn_size, keep_prob) for _ in range(hp.num_layers)])
            dec_att_cell = self.make_attention_cell(decoder_cell, hp.rnn_size, self.encoder_outputs, seq_lengths)
            dec_init_state = dec_att_cell.zero_state(batch_size, dtype=tf.float32).clone(cell_state=self.encoder_state)
            output_layer = Dense(vocab_size, name='output_projection')

            if is_training:
                helper = tf.contrib.seq2seq.TrainingHelper(self.dec, self.y_lengths)
                basic_decoder = tf.contrib.seq2seq.BasicDecoder(dec_att_cell,
                                                                helper,
                                                                dec_init_state,
                                                                output_layer=output_layer)
                outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(basic_decoder,
                                                                                    output_time_major=False,
                                                                                    maximum_iterations=maxlen,
                                                                                    swap_memory=False,
                                                                                    impute_finished=True)
                dec_output = outputs.rnn_output
                self.sample_id = outputs.sample_id

                self.logits = tf.concat(
                    [dec_output, tf.zeros([batch_size, maxlen - tf.shape(dec_output)[1], vocab_size])], axis=1)

            else:
                start_tokens = tf.fill([tf.shape(enc_inputs)[0]], 2)

                if hp.beam_width == 0:

                    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(word_emb, start_tokens, 3)
                    basic_decoder = tf.contrib.seq2seq.BasicDecoder(dec_att_cell,
                                                                    helper,
                                                                    dec_init_state,
                                                                    output_layer=output_layer)
                    outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(basic_decoder,
                                                                                        output_time_major=False,
                                                                                        maximum_iterations=maxlen,
                                                                                        swap_memory=False,
                                                                                        impute_finished=False)

                    dec_output = outputs.rnn_output
                    self.sample_id = outputs.sample_id
                    self.logits = dec_output

                else:
                    beam_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                        cell=dec_att_cell,
                        embedding=word_emb,
                        start_tokens=start_tokens,
                        end_token=3,
                        initial_state=dec_init_state,
                        beam_width=hp.beam_width,
                        output_layer=output_layer
                    )

                    outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(beam_decoder,
                                                                                        output_time_major=False,
                                                                                        maximum_iterations=maxlen,
                                                                                        swap_memory=False,
                                                                                        impute_finished=False)
                    self.logits = tf.no_op()
                    self.sample_id = outputs.predicted_ids

        if is_training:
            with tf.variable_scope("loss"):
                target_weights = tf.sequence_mask(self.y_lengths, maxlen, dtype=tf.float32, name="mask")
                self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.logits,
                                                targets=dec_targets,
                                                weights=target_weights,
                                                average_across_timesteps=True,
                                                average_across_batch=True)
                self.word_count = tf.reduce_sum(seq_lengths) + tf.reduce_sum(self.y_lengths)
                self.predict_count = tf.reduce_sum(self.y_lengths)
                self.global_step = tf.Variable(0, trainable=False)

                # Optimizer
                opt = tf.train.AdamOptimizer(hp.lr)

                # Gradients
                if hp.clip > 0:
                    grads, vs = zip(*opt.compute_gradients(self.loss))
                    grads, gnorm = tf.clip_by_global_norm(grads, hp.clip)
                    self.train_op = opt.apply_gradients(zip(grads, vs), global_step=self.global_step)

        with tf.name_scope("misc"):
            self.saver = tf.train.Saver()
            self.init = tf.global_variables_initializer()


    def make_rnn_cell(self, rnn_size, keep_prob):
        cell = tf.nn.rnn_cell.LSTMCell(rnn_size)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_prob)
        return cell

    def make_attention_cell(self, dec_cell, rnn_size, enc_output, lengths, alignment_history=False):
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=rnn_size,
                                                                   memory=enc_output,
                                                                   memory_sequence_length=lengths,
                                                                   normalize= True,
                                                                   name='BahdanauAttention')

        return tf.contrib.seq2seq.AttentionWrapper(cell=dec_cell,
                                                   attention_mechanism=attention_mechanism,
                                                   attention_layer_size=None,
                                                   output_attention=False,
                                                   alignment_history=alignment_history)

    def train(self):

        enc_input, dec_input = load_train_data("transformer", hp.datapath)
        num_batch = enc_input.shape[0] // hp.batch_size

        with self.graph.as_default():
            self.build_graph(is_training=True)

        with tf.Session(graph=self.graph) as sess:

            # load from checkpoint
            if tf.train.get_checkpoint_state(hp.logdir_s):
                self.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir_s))
                a = re.findall("[0-9]+", tf.train.latest_checkpoint(hp.logdir_s))
                start_epoch, global_step = re.findall("[0-9]+", tf.train.latest_checkpoint(hp.logdir_s))
                start_epoch = int(start_epoch)+1
                print('Model restored')

                if hp.num_epochs_trans > start_epoch:
                    print("Starting from epoch : ", start_epoch)

            # or initialize
            else:
                sess.run(self.init)
                start_epoch = 1
                print("Starting from scratch")

            for epoch in range(start_epoch, hp.num_epochs_trans+1):

                epoch_loss = 0

                # initialize iterator
                sess.run(self.iter.initializer)
                for step in range(num_batch):

                    _, step_loss = sess.run([self.train_op, self.loss])
                    epoch_loss += step_loss

                print("Current Epoch : {:02d} Loss: {:.4f}".format(epoch, epoch_loss/num_batch))
                curr_gstep = sess.run(self.global_step)

                if epoch % 5 == 0:
                    self.saver.save(sess, hp.logdir_s + '/model_epoch_{:02d}_gs_{:d}' .format(epoch, curr_gstep))

            print("Training Done!")



