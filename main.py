#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from data_utils import get_trimmed_glove_vectors, load_vocab, \
    get_processing_word, ds
from general_utils import get_logger
from model import sentiModel
from config import config

# directory for training outputs
if not os.path.exists(config.output_path):
    os.makedirs(config.output_path)

# load vocabs
vocab_words = load_vocab(config.words_filename)
vocab_tags  = load_vocab(config.tags_filename)


# get processing functions
processing_word = get_processing_word(vocab_words,
                lowercase=config.lowercase)
processing_tag  = get_processing_word(vocab_tags, lowercase=False)

# get pre trained embeddings

embeddings = get_trimmed_glove_vectors(config.trimmed_filename)

# create dataset
dev   = ds(config.dev_filename, processing_word,
                    processing_tag, config.max_iter)
test  = ds(config.test_filename, processing_word,
                    processing_tag, config.max_iter)
train = ds(config.train_filename, processing_word,
                    processing_tag, config.max_iter)

# get logger
logger = get_logger(config.log_path)

# build model
model = sentiModel(config, embeddings, nlabels=len(vocab_tags),
                 logger=logger)
model.build()

# train, evaluate and interact
model.train(train, dev)
#model.evaluate(test, vocab_tags)
#model.interactive_shell(vocab_tags, processing_word)
