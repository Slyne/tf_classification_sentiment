#!/usr/bin/env python
# -*- coding: utf-8 -*-


import theano
from keras.utils.np_utils import to_categorical
from data_utils import get_trimmed_glove_vectors, load_vocab, get_processing_word,ds,minibatches, pad_sequences
from config import *
import numpy as np
from keras.models import load_model



def pad_embedding(trimmed_filename, words_filename):
    embeddings = get_trimmed_glove_vectors(trimmed_filename)
    (rows,columns) = embeddings.shape
    add_row = np.zeros((1,columns))
    new_embedding = np.concatenate((add_row,embeddings),axis=0)
    vocab = load_vocab(words_filename)
    for word in vocab:
        vocab[word] += 1
    return (new_embedding, vocab)





print config
embedding, vocab_words = pad_embedding(config.trimmed_filename, config.words_filename)

vocab_tags = load_vocab(config.tags_filename)

processing_word = get_processing_word(vocab_words,
                lowercase=config.lowercase)
processing_tag  = get_processing_word(vocab_tags, lowercase=False)


# create dataset
train  = ds(config.train_filename, processing_word,
                    processing_tag, config.max_iter)


from keras.models import Model
org_model = load_model("result/model.weights.binary/acc_0.833655714989.model")
#from keras.utils.visualize_util import plot
#plot(org_model, to_file='model.png')

print org_model.layers[1].get_weights()[0].shape
print org_model.layers[2].get_weights()[0].shape


layer_name = 'bidirectional_1'
intermediate_layer_model = Model(input=org_model.input,   # keras 2.0  inputs  outputs
                                 output=org_model.get_layer(layer_name).output)


out = open("layer_output", "w")
for i, (words, labels) in enumerate(minibatches(train, config.batch_size)):
    word_ids, sequence_lengths = pad_sequences(words, 0, config.maximum_sequence_length)
    intermediate_output = intermediate_layer_model.predict(np.asarray(word_ids))
    for (layer_output, label) in zip(intermediate_output,labels):
        out.write(str(label) + "\t")
        out_layer = '\t'.join(str(x) for x in layer_output)
        out.write(out_layer + "\n")
