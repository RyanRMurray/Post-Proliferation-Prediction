import json
import cv2
import re
import math
import itertools
import sys
import os

from tensorflow.python.keras.layers.recurrent import SimpleRNN
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, Flatten, Embedding, LSTM, ReLU, BatchNormalization, Lambda, Concatenate, Reshape, concatenate, SimpleRNN, TimeDistributed
from tensorflow.keras.regularizers import L2
from tensorflow.keras.models import Sequential
from keras.utils.vis_utils import plot_model

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
def pre_joint_embed_layers(inputs, fc1,fc2):
    pre_joint = Dense(fc1)(inputs)
    pre_joint = ReLU()(pre_joint)
    pre_joint = Dense(fc2)(pre_joint)
    pre_joint = BatchNormalization()(pre_joint)
    pre_joint = Lambda(lambda x: tf.keras.backend.l2_normalize(x,axis=1))(pre_joint)
    
    return pre_joint

def lstm_branch(word_num, input_size):
    lstm = Sequential()
    embedding = Embedding(word_num, 256 , input_length=input_size, name='lstm_embedder')
    lstm.add(embedding)
    lstm.add(LSTM(256))

    t_branch = tf.keras.Model(inputs= lstm.input, outputs=pre_joint_embed_layers(lstm.output,512,256))

    #t_branch.summary()
    print("Generated lstm branch")
    
    return t_branch

def cnn_branch():
    inceptionresnet = tf.keras.applications.InceptionResNetV2(include_top=False)
    
    for layer in inceptionresnet.layers:
        layer.trainable = False

    pre_joint = pre_joint_embed_layers(inceptionresnet.output,768,256)

    #reshape
    reshaped = Reshape((256,))(pre_joint)

    i_branch = tf.keras.Model(inputs=inceptionresnet.input, outputs=reshaped)
    
    #i_branch.summary()
    print("Generated cnn branch")
    return i_branch

def detail_input():
    #details:
    #   day-of-week of post,
    #   time-of-day of post,
    #   author account age,
    #   author followers count,
    #   author following count,
    #   author tweet count,
    #   author in list count,
    #   author verified status
    calc_detail_vector_size = lambda x : 2 + ((2**x)*x) + x - (2**(x+1))
    input_size = calc_detail_vector_size(8)

    d_branch = Input(shape=(input_size,))

    print("Generated details branch")
    return d_branch



def build_model(data_dir : str):
    data = json.load(open(data_dir, 'r'))

    #generate sequences
    (data, word_count, text_input_length) = tokenize_data(data)
    
    #generate branches
    i_branch = cnn_branch()
    t_branch = lstm_branch(word_count, text_input_length)
    d_branch = detail_input()
    #joint embedding and convolution
    joint = concatenate([i_branch.output, t_branch.output])
    joint = Dense(256)(joint)
    joint = Dense(128)(joint)
    
    #add details input
    joint = concatenate([joint, d_branch])
    model = Dense(128)(joint)
    
    rnn_in = Input(shape=(None,1))
    rnn = SimpleRNN(128, return_sequences=True, return_state=True)
    dist = TimeDistributed(Dense(1, activation='relu'))


    model, _ = rnn(rnn_in, initial_state=model)
    model = dist(model)

    final = tf.keras.Model(inputs=[i_branch.input, t_branch.input, d_branch, rnn_in], outputs=model)
    print("Generated model")
    return final
    #todo: joint embedding constraint function, save model and tokenizer

def main():
    #check input then load
    if len(sys.argv) != 2:
        print("Run using the following command:\n\tpython3 prediction_network.py <data set path>")
        return
    
    data_dir = sys.argv[1]

    if data_dir[-5:] != '.json' or not os.path.isfile(data_dir):
        print("Please enter a path to a valid json file")
        return

    model = build_model(data_dir)
    
    plot_model(model, to_file='model_plot.png', show_shapes=True)

main()

'''
i = cv2.imread("./Data/200ktweets_Images/x.jpg")
i = tf.expand_dims(i, axis=0)

print(i_branch.predict(i))
'''