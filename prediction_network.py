import json
import cv2
import re
import math
import itertools
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Flatten, Embedding, LSTM, GRU
from tensorflow.keras.models import Sequential

from typing import Tuple

#create token sequences for our inputs, and get info for generating the lstm branch
def tokenize_data(data : list) -> Tuple[list, int, int]:
    tweets = len(data)
    words = set()
    input_size = 0

    for tweet in data:
        symbols = re.split(r'\s+', tweet['text'])
        input_size = max(input_size, len(symbols))
        words.update(symbols)

    dims = math.ceil(len(words) ** (1/4))

    print('{} unique symbols, input size is {}. Using {} dimensions.'.format(len(words), input_size, dims))

    tokenizer = Tokenizer(len(words), filters='!"#$%&()*+,-/:;=?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts([tweet['text'] for tweet in data])

    counter = 0
    for tweet in data:
        tokens = np.array(
            tokenizer.texts_to_sequences([tweet['text']])[0]
        )
        #we pad the left side like this because we're iterating on each json object
        seq = np.zeros(input_size)
        seq[-len(tokens):] = tokens
        tweet['sequence'] = seq

        counter += 1
        print('Generated {}/{} sequences'.format(counter, tweets), end='\r')

    print()
    print("Generated Sequences")

    return (data, len(words), input_size)


def lstm_branch(word_num, input_size):
    t_branch = Sequential()
    embedding = Embedding(word_num, 32, input_length=input_size, name='lstm_embedder')
    t_branch.add(embedding)
    t_branch.add(LSTM(32))

    print("Generated lstm branch")
    t_branch.summary()
    return t_branch

def cnn_branch():
    inceptionresnet = tf.keras.applications.InceptionResNetV2()
    inception_output = inceptionresnet.layers[-2].output
    i_branch = tf.keras.Model(inputs = inceptionresnet.input, outputs = inception_output)

    print("Generated cnn branch")

    return i_branch

def main():
    #check input then load
    if len(sys.argv) != 2:
        print("Run using the following command:\n\tpython3 prediction_network.py <data set path>")
        return
    
    data_dir = sys.argv[1]

    if data_dir[-5:] != '.json' or not os.path.isfile(data_dir):
        print("Please enter a path to a valid json file")
        return

    data = json.load(open(data_dir, 'r'))

    #generate sequences
    (data, word_count, text_input_length) = tokenize_data(data)

    #generate branches
    i_branch = cnn_branch()
    t_branch = lstm_branch(word_count, text_input_length)   

main()

'''
inceptionresnet = tf.keras.applications.InceptionResNetV2()
inception_output = inceptionresnet.layers[-2].output
i_branch = tf.keras.Model(inputs = inceptionresnet.input, outputs = inception_output)

i = cv2.imread("./Data/200ktweets_Images/1408753241932845057.jpg")
i = tf.expand_dims(i, axis=0)

print(i_branch.predict(i))
'''