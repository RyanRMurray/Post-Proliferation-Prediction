import json
import cv2
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
    words = set()
    input_size = 0

    
    return



'''
inceptionresnet = tf.keras.applications.InceptionResNetV2()
inception_output = inceptionresnet.layers[-2].output
i_branch = tf.keras.Model(inputs = inceptionresnet.input, outputs = inception_output)

i = cv2.imread("./Data/200ktweets_Images/1408753241932845057.jpg")
i = tf.expand_dims(i, axis=0)

print(i_branch.predict(i))
'''