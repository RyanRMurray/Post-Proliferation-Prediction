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
from tensorflow.keras.layers import Dense, Flatten, Embedding, LSTM, GRU, ReLU, BatchNormalization, Lambda
from tensorflow.keras.regularizers import L2
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

#adds layers as described in rt wars paper
def pre_joint_embed_layers(inputs, units):
    pre_joint = Dense(units)(inputs)
    pre_joint = ReLU()(pre_joint)
    pre_joint = Dense(units)(pre_joint)
    pre_joint = BatchNormalization()(pre_joint)
    pre_joint = Lambda(lambda x: tf.keras.backend.l2_normalize(x,axis=1))(pre_joint)
    
    return pre_joint

def lstm_branch(word_num, input_size):
    lstm = Sequential()
    embedding = Embedding(word_num, 32, input_length=input_size, name='lstm_embedder')
    lstm.add(embedding)
    lstm.add(LSTM(32))

    t_branch = tf.keras.Model(inputs= lstm.input, outputs=pre_joint_embed_layers(lstm.output,32))

    #t_branch.summary()
    print("Generated lstm branch")
    
    return t_branch

def cnn_branch():
    inceptionresnet = tf.keras.applications.InceptionResNetV2(include_top=False)
    
    for layer in inceptionresnet.layers:
        layer.trainable = False

    i_branch = tf.keras.Model(inputs=inceptionresnet.input, outputs=pre_joint_embed_layers(inceptionresnet.output,1536))
    
    #i_branch.summary()
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