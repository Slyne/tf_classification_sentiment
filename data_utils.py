#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
UNK = "$UNK$"


class ds(object):
    def __init__(self, filename, processing_word=None, processing_tag=None,max_iter=None):
        self.filename = filename
        self.processing_word = processing_word
        self.processing_tag = processing_tag
        self.max_iter = max_iter
        self.length = None




    def __iter__(self):

        with open(self.filename) as f:
            n_iter = 0
            for line in f:
                line = line.decode("utf-8")
                sline = line.strip().split()
                label = sline[0:1]
                words = sline[1:]
                if self.processing_word is not None:
                    words = [ self.processing_word(word) for word in words]
                if self.processing_tag is not None:
                    label = [self.processing_tag(l) for l in label]
                yield words, label
                n_iter +=1
                if self.max_iter is not None and n_iter > self.max_iter:
                    break

    def __len__(self):
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1
        return self.length


def get_vocabs(datasets):
    """
    Args:
        datasets: a list of dataset objects
    Return:
        a set of all the words in the dataset
    """
    print ("Building vocab...")
    vocab_words = set()
    vocab_tags = set()
    for dataset in datasets:
        for words, label in dataset:
            vocab_words.update(words)
            vocab_tags.update(label)
    print ("- done. {} tokens".format(len(vocab_words)))
    return vocab_words, vocab_tags

def get_glove_vocab(filename):
    """
    Args:
        filename: path to the glove vectors
    """
    print ("Building vocab...")
    vocab = set()
    with open(filename) as f:
        for line in f:
            line = line.decode("utf-8")
            word = line.strip().split(' ')[0]
            vocab.add(word)
    print ("- done. {} tokens".format(len(vocab)))
    return vocab

def write_vocab(vocab, filename):
    """
    Writes a vocab to a file
    Args:
        vocab: iterable that yields word
        filename: path to vocab file
    Returns:
        write a word per line
    """
    print("Writing vocab...")
    with open(filename, "w") as f:
        for i, word in enumerate(vocab):
            word = word.encode("utf-8")
            if i != len(vocab) - 1:
                f.write("{}\n".format(word))
            else:
                f.write(word)
    print ("- done. {} tokens".format(len(vocab)))

def load_vocab(filename):
    """
    Args:
        filename: file with a word per line
    Returns:
        d: dict[word] = index
    """
    d = dict()
    with open(filename) as f:
        for idx, word in enumerate(f):
            word = word.strip().decode("utf-8")
            d[word] = idx
    return d


def export_trimmed_glove_vectors(vocab, glove_filename, trimmed_filename, dim):
    """
    Saves glove vectors in numpy array

    Args:
        vocab: dictionary vocab[word] = index
        glove_filename: a path to a glove file
        trimmed_filename: a path where to store a matrix in npy
        dim: (int) dimension of embeddings
    """
    embeddings = np.zeros([len(vocab), dim])
    with open(glove_filename) as f:
        for line in f:
            line = line.decode("utf-8")
            line = line.strip().split()
            word = line[0]
            embedding = map(float, line[1:])
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(list(embedding))
    np.savez_compressed(trimmed_filename, embeddings=embeddings)


def get_trimmed_glove_vectors(filename):
    """
    Args:
        filename: path to the npz file
    Returns:
        matrix of embeddings (np array)
    """
    print(filename)
    #with open(filename) as f:
    return np.load(filename)["embeddings"]


def get_processing_word(vocab_words=None,
                            lowercase=False):

    """
    Args:
        vocab: dict[word] = idx
    Returns:
        f("cat") = ([12, 4, 32], 12345)
                 = (list of char ids, word id)
    """


    def f(word):
        # 1. preprocess word
        if lowercase:
            word = word.lower()

        # 2. get id of word
        if vocab_words is not None:
            if word in vocab_words:
                word = vocab_words[word]
            else:
                word = vocab_words[UNK]

        # 3.  word id
        return word

    return f



def _pad_sequences(sequences, pad_tok, max_length):
    """
        Args:
            sequences: a generator of list or tuple
            pad_tok: the char to pad with
        Returns:
            a list of list where each sublist has same length
        """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, maximum=None):
    """
        Args:
            sequences: a generator of list or tuple
            pad_tok: the char to pad with
        Returns:
            a list of list where each sublist has same length
        """
    max_length = max(map(lambda x: len(x), sequences))
    if maximum is not None:
        max_length = min(max_length,maximum)
    sequence_padded, sequence_length = _pad_sequences(sequences,
                                                      pad_tok, max_length)
    return sequence_padded, sequence_length

def minibatches(data, minibatch_size):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)
    Returns:
        list of tuples
    """
    x_batch, y_batch = [], []
    for (x, y) in data:
        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []
        x_batch += [x]
        y_batch += [y[0]]

    if len(x_batch) != 0:
        yield x_batch, y_batch