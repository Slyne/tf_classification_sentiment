#!/usr/bin/env python
# -*- coding: utf-8 -*-
#!/usr/bin/env python
# -*- coding: utf-8 -*-
class config():
    dim = 200  # embedding dimension

    glove_filename = "../data/embedding/embedding.model".format(dim)
    trimmed_filename = "../data/embedding{}d.trimmed.npz".format(dim)
    words_filename = "../data/words.txt"
    tags_filename = "../data/tags.txt"

    dev_filename = "../data/raw_saraba1st/train_data_all.val"
    test_filename = "../data/raw_saraba1st/train_data_all.test"
    train_filename = "../data/raw_saraba1st/train_data_all.train"
    max_iter = None
    lowercase = True
    train_embeddings = False
    nepochs = 20
    dropout = 0.5
    batch_size = 10
    maximum_sequence_length = 50
    #lr = 0.001
    lr = 0.0001
    lr_decay = 0.9
    nepoch_no_imprv = 3
    hidden_size = 100
    crf = True # if crf, training is 1.7x slower
    output_path = "result/"
    model_output = output_path + "model.weights.binary/"
    log_path = output_path + "log.txt"