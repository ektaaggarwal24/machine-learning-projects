import numpy as np
#from random import sample
import random
import config

def split_data(x, y, ratio=[0.8, 0.1, 0.1]):
    data_length = len(x)
    lengths = [int(data_length * item) for item in ratio]

    x_train, y_train = x[:lengths[0]], y[:lengths[0]]
    x_test, y_test = x[lengths[0]:lengths[0]+lengths[1]], y[lengths[0]:lengths[0]+lengths[1]]
    x_valid, y_valid = x[-lengths[-1]:], y[-lengths[-1]:]

    return (x_train, y_train), (x_test, y_test), (x_valid, y_valid)

def generate_random_batch(x, y, batch_size):
    while True:
        #sample_idx = sample(list(np.arange(len(x))), batch_size)
        sample_idx = [random.choice(list(np.arange(len(x)))) for i in range(config.batch_size)]
        #yield x[sample_idx].T, y[sample_idx].T
        yield [x[i] for i in sample_idx], [y[i] for i in sample_idx]

def decode(sequence, lookup, separator=''):
    return separator.join([lookup[element] for element in sequence if element])