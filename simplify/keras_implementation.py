#!/usr/bin/env python
# -*- coding: utf-8 -*-
import keras
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional,Dense, Activation,LSTM
from keras.utils.np_utils import to_categorical
from data_utils import get_trimmed_glove_vectors, load_vocab, get_processing_word,ds,minibatches, pad_sequences
from config import *
import numpy as np
from keras.utils.generic_utils import Progbar
from keras.models import load_model
import glob
import os


def pad_embedding(trimmed_filename, words_filename):
    embeddings = get_trimmed_glove_vectors(trimmed_filename)
    (rows,columns) = embeddings.shape
    add_row = np.zeros((1,columns))
    new_embedding = np.concatenate((add_row,embeddings),axis=0)
    vocab = load_vocab(words_filename)
    for word in vocab:
        vocab[word] += 1
    return (new_embedding, vocab)


def create_model_classify(max_features, nlabels, embeddings=None, embedding_dim=200, hidden_dim=300):
    model = Sequential()
    if embeddings is not None:
        model.add(Embedding(input_dim=max_features,output_dim=embedding_dim, dropout=0.5, weights=[embeddings], mask_zero=True))
    else:
        model.add(Embedding(max_features, embedding_dim=embedding_dim, dropout=0.5))
    model.add(Bidirectional(LSTM(hidden_dim, dropout_W=0.2, dropout_U=0.2, return_sequences=False)))  # try using a GRU instead, for fun
    model.add(Dense(nlabels))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy',]) # availabel metrics https://keras.io/metrics/
    return model


def create_model_regress(max_features, embeddings=None, embedding_dim=200, hidden_dim=300):
    model = Sequential()
    if embeddings is not None:
        model.add(Embedding(input_dim=max_features,output_dim=embedding_dim, dropout=0.2, weights=[embeddings], mask_zero=True))
    else:
        model.add(Embedding(max_features, embedding_dim=embedding_dim, dropout=0.2))
    model.add(
        Bidirectional(LSTM(hidden_dim, dropout_W=0.2, dropout_U=0.2),
                      merge_mode='concat'))  # try using a GRU instead, for fun

    model.compile(loss='mean_squared_error',
                  optimizer='sgd',
                  metrics=['mae','acc' ])  # availabel metrics https://keras.io/metrics/idden_dim=
    return model




embedding, vocab_words = pad_embedding(config.trimmed_filename, config.words_filename)

vocab_tags = load_vocab(config.tags_filename)

processing_word = get_processing_word(vocab_words,
                lowercase=config.lowercase)
processing_tag  = get_processing_word(vocab_tags, lowercase=False)


# create dataset
dev   = ds(config.dev_filename, processing_word,
                    processing_tag, config.max_iter)
test  = ds(config.test_filename, processing_word,
                    processing_tag, config.max_iter)
train = ds(config.train_filename, processing_word,
                    processing_tag, config.max_iter)

model = create_model_classify(len(vocab_words)+1,len(vocab_tags), embeddings=embedding, hidden_dim=config.hidden_size)


if not os.path.exists(config.model_output):
    os.makedirs(config.model_output)
else:
    list_of_files = glob.glob(config.model_output+"/*")
    if len(list_of_files) != 0:
        latest_file = max(list_of_files, key=os.path.getctime)
        print("load model.....{}".format(latest_file))
        model = load_model(latest_file)


nbatches = (len(train) + config.batch_size - 1) / config.batch_size
best_score = 0
niter = 0
for j in range(config.nepochs):
    progbar = Progbar(target=nbatches)
    print("epoch_{}:".format(j))
    for i, (words, labels) in enumerate(minibatches(train, config.batch_size)):
        word_ids, sequence_lengths = pad_sequences(words, 0, config.maximum_sequence_length)
        labels = to_categorical(labels,len(vocab_tags)).astype("int")
        loss,acc = model.train_on_batch(np.asarray(word_ids), np.asarray(labels))
        progbar.add(1, values=[("train loss", loss), ("acc", acc)])

    words, labels = list(minibatches(test,len(test)))[0]
    word_ids, sequence_lengths = pad_sequences(words, 0, config.maximum_sequence_length)
    labels = to_categorical(labels, len(vocab_tags)).astype("int")
    val_loss, val_acc = model.test_on_batch(word_ids, labels)
    print("val_loss: {} val_acc:{}".format(val_loss, val_acc))
    best_score = 0
    if val_acc >= best_score: # early stop
        best_score = val_acc
        # save best model
        model.save(config.model_output+"acc_"+str(best_score)+".model")
    else:
        niter += 1
        if niter == config.nepoch_no_imprv:
            break
