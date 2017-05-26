#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/env python
# -*- coding: utf-8 -*-
from config import config
from data_utils import ds, get_vocabs, UNK, \
    get_glove_vocab, write_vocab, load_vocab, \
    export_trimmed_glove_vectors, get_processing_word


def build_data(config):
    """
    Procedure to build data
    Args:
        config: defines attributes needed in the function
    Returns:
        creates vocab files from the datasets
        creates a npz embedding file from trimmed glove vectors
    """
    #processing_word = get_processing_word(lowercase=config.lowercase)
    # Generators
    dev   = ds(config.dev_filename)
    test  = ds(config.test_filename)
    train = ds(config.train_filename)

    # Build Word and Tag vocab
    vocab_words, vocab_tags = get_vocabs([train, dev, test])
    vocab_glove = get_glove_vocab(config.glove_filename)

    vocab = vocab_words & vocab_glove
    vocab.add(UNK)


    # Save vocab
    write_vocab(vocab, config.words_filename)
    write_vocab(vocab_tags, config.tags_filename)

    # Trim GloVe Vectors
    vocab = load_vocab(config.words_filename)
    export_trimmed_glove_vectors(vocab, config.glove_filename,
                                config.trimmed_filename, config.dim)



if __name__ == "__main__":
    build_data(config)