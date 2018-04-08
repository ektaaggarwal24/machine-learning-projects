import tensorflow as tf
import numpy as np
import data_processing
import config
import data_utils
import seq2seq_wrapper
from os import path
import sys

#load data and split into train and test sets
idx_headings, idx_descriptions = data_processing.process_data()
article_metadata = data_processing.unpickle_articles("CNN_article_data.pkl")
(x_train, y_train ), (x_test, y_test), (x_valid, y_valid) = data_utils.split_data(idx_descriptions, idx_headings)

#define parameters
if (sys.argv[1]).lower() == "train":
    xseq_length = len(x_train)
    yseq_length = len(y_train)
else:
    xseq_length = len(x_test)
    yseq_length = len(y_test)

batch_size = config.batch_size
xvocab_size = len(article_metadata['idx2word'])
yvocab_size = xvocab_size
checkpoint_path = path.join(config.path_outputs)

print (checkpoint_path)

#define model
model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_length,
                                yseq_len=yseq_length,
                                xvocab_size=xvocab_size,
                                yvocab_size=yvocab_size,
                                emb_dim=config.embedding_dim,
                                num_layers=3,
                                ckpt_path=checkpoint_path)

val_batch_gen = data_utils.generate_random_batch(x_valid, y_valid, config.batch_size)
train_batch_gen = data_utils.generate_random_batch(x_train, y_train, config.batch_size)


isSessionRestored = model.restore_last_session()
if (sys.argv[1]).lower() == "train":
    model.train(train_batch_gen, val_batch_gen,isSessionRestored)
else:
    result = model.predict(x_test)
    for i in result:
        print(x_test[i])




