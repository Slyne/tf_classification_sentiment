#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import os
import tensorflow as tf
import logging
from data_utils import pad_sequences, minibatches
from general_utils import Progbar

class sentiModel(object):
    def __init__(self, config, embeddings, nlabels, logger=None):
        self.config = config
        self.embeddings = embeddings
        self.nlabels = nlabels
        self.logger = logger
        if logger is None:
            logger = logging.getLogger('logger')
            logger.setLevel(logging.DEBUG)
            logging.basciConfig(format='%(message)s', lebel=logging.DEBUG)

    def add_placeholders(self):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")# batch size, max length of sentence in batch
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_length")# shape = batch size
        self.labels = tf.placeholder(tf.int32, shape=[None],name="labels") #only one dimension
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                                      name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")


    def get_feed_dict(self, words, labels=None, lr=None, dropout=None):
        word_ids, sequence_lengths = pad_sequences(words, 0)
        feed = {
            self.word_ids: word_ids,
            self.sequence_lengths: sequence_lengths
        }
        if labels is not None:
            feed[self.labels] = labels

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed, sequence_lengths


    def add_word_embeddings_op(self):
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(self.embeddings, name="_word_embeddings", dtype=tf.float32,
                                           trainable=self.config.train_embeddings)
            word_embeddings = tf.nn.embedding_lookup(_word_embeddings, self.word_ids,
                                                     name="word_embeddings")
        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)


    def add_logits_op(self):
        """
        Adds logits to self
        """
        with tf.variable_scope("bi-lstm"):
            lstm_cell = tf.contrib.rnn.LSTMCell(self.config.hidden_size)
            _, (output_state_fw, output_state_bw) = tf.nn.bidirectional_dynamic_rnn(lstm_cell,
                                                                        lstm_cell, self.word_embeddings,
                                                                        sequence_length=self.sequence_lengths,
                                                                        dtype=tf.float32)
            output = tf.concat((output_state_fw[-1], output_state_bw[-1]), axis=-1)
            #self.output = output
            output = tf.nn.dropout(output, self.dropout)


        with tf.variable_scope("proj"):
            W = tf.get_variable("W", shape=[2 * self.config.hidden_size, self.nlabels],
                                dtype=tf.float32)

            b = tf.get_variable("b", shape=[self.nlabels], dtype=tf.float32,
                                initializer=tf.zeros_initializer())

            output = tf.reshape(output, [-1, 2 * self.config.hidden_size])

            pred = tf.matmul(output, W) + b
            self.logits = pred


    def add_pred_op(self):
        """
        Adds labels_pred to self
        """
        self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)


    def add_loss_op(self):
        """
        Adds loss to self
        """
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
        #self.loss = losses
        self.loss = tf.reduce_mean(losses)

        # for tensorboard
        tf.summary.scalar("loss", self.loss)

    def add_train_op(self):
        """
        Add train_op to self
        """
        with tf.variable_scope("train_step"):
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.minimize(self.loss)

    def add_init_op(self):
        self.init = tf.global_variables_initializer()

    def add_summary(self, sess):
        # tensorboard stuff
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.output_path, sess.graph)

    def build(self):
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_logits_op()
        self.add_pred_op()
        self.add_loss_op()
        self.add_train_op()
        self.add_init_op()

    def predict_batch(self, sess, words):
        fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0)
        labels_pred = sess.run(self.labels_pred, feed_dict=fd)

        return labels_pred


    def run_epoch(self, sess, train, dev, epoch):
        """
        Performs one complete pass over the train set and evaluate on dev
        Args:
            sess: tensorflow session
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            tags: {tag: index} dictionary
            epoch: (int) number of the epoch
        """
        nbatches = (len(train) + self.config.batch_size - 1) / self.config.batch_size
        prog = Progbar(target=nbatches)
        for i, (words, labels) in enumerate(minibatches(train, self.config.batch_size)):
            fd, _ = self.get_feed_dict(words, labels, self.config.lr, self.config.dropout)

            _, train_loss, summary, hidden = sess.run([self.train_op, self.loss, self.merged], feed_dict=fd)
            prog.update(i + 1, [("train loss", train_loss)])

            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch * nbatches + i)

        acc, f1 = self.run_evaluate(sess, dev)
        self.logger.info("- dev acc {:04.2f} - f1 {:04.2f}".format(100 * acc, 100 * f1))
        return acc, f1

    def run_evaluate(self, sess, test):
        """
        Evaluates performance on test set
        Args:
            sess: tensorflow session
            test: dataset that yields tuple of sentences, tags
            tags: {tag: index} dictionary
        Returns:
            accuracy
            f1 score
        """
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for words, labels in minibatches(test, self.config.batch_size):
            labels_pred = self.predict_batch(sess, words)
            lab = labels
            lab_pred = labels_pred
            accs += map(lambda element: element[0] == element[1], zip(lab, lab_pred))
            lab = set(lab)
            lab_pred = set(lab_pred)
            correct_preds += len(lab & lab_pred)
            total_preds += len(lab_pred)
            total_correct += len(lab)

        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)
        return acc, f1

    def train(self, train, dev):
        """
        Performs training with early stopping and lr exponential decay
        Args:
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            tags: {tag: index} dictionary
        """
        best_score = 0
        saver = tf.train.Saver()
        # for early stopping
        nepoch_no_imprv = 0
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(self.config.model_output)
            # restore session
            if ckpt and ckpt.model_checkpoint_path:
                print(ckpt)
                saver.restore(sess, self.config.model_output)
            else:
                print("Begin to initialize ...")
                sess.run(self.init)
            # tensorboard
            self.add_summary(sess)
            for epoch in range(self.config.nepochs):
                self.logger.info("Epoch {:} out of {:}".format(epoch + 1, self.config.nepochs))

                acc, f1 = self.run_epoch(sess, train, dev, epoch)

                # decay learning rate
                self.config.lr *= self.config.lr_decay

                # early stopping and saving best parameters
                if f1 >= best_score:
                    nepoch_no_imprv = 0
                    if not os.path.exists(self.config.model_output):
                        os.makedirs(self.config.model_output)
                    saver.save(sess, self.config.model_output)
                    best_score = f1
                    self.logger.info("- new best score!")

                else:
                    nepoch_no_imprv += 1
                    if nepoch_no_imprv >= self.config.nepoch_no_imprv:
                        self.logger.info("- early stopping {} epochs without improvement".format(
                            nepoch_no_imprv))
                        break

    def evaluate(self, test):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            self.logger.info("Testing model over test set")
            saver.restore(sess, self.config.model_output)
            acc, f1 = self.run_evaluate(sess, test)
            self.logger.info("- test acc {:04.2f} - f1 {:04.2f}".format(100 * acc, 100 * f1))


    def interactive_shell(self,  processing_word):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.config.model_output)
            self.logger.info("This is an interactive mode, enter a sentence:")
            while True:
                try:
                    sentence = input("input> ")
                    # words_raw = sentence.strip().split(" ")  english words seperated by space
                    words_raw = list(sentence.strip())
                    # print(words_raw)
                    words = map(processing_word, words_raw)
                    words = list(words)
                    # if type(words[0]) == tuple:
                    #    words = zip(*words)
                    preds= self.predict_batch(sess, [words])[0]
                    print(words_raw)
                    print(preds)
                except EOFError:
                    print("Closing session.")
                    break