import json
import cv2
import re
import math
import itertools
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

def word_embedding(data : json):
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

    return data

j = json.load(open('./Data Sets/200ktweets.json', 'r'))

for (_,x) in zip(range(10), word_embedding(j)):
    print(x['text'])
    print(x['sequence'])
    print()


'''
inceptionresnet = tf.keras.applications.InceptionResNetV2()
inception_output = inceptionresnet.layers[-2].output
i_branch = tf.keras.Model(inputs = inceptionresnet.input, outputs = inception_output)

i = cv2.imread("./Data/200ktweets_Images/1408753241932845057.jpg")
i = tf.expand_dims(i, axis=0)

print(i_branch.predict(i))
'''