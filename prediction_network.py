import json
import cv2
import re
import math
import itertools
import sys
from datetime import datetime
from PIL import Image

import os

from tensorflow.keras import regularizers
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import argparse
parser = argparse.ArgumentParser(description='Generate and train a model to predict success of Twitter posts')
parser.add_argument('dataset', metavar='dataset', type=str, help='Path to the training data set')
parser.add_argument('model', metavar='model', type=str, nargs='?', help='Path to a pre-generated model (optional)', default=None)

from tensorflow.python.keras.layers.recurrent import SimpleRNN
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

#details:
#   day-of-week of post,
#   time-of-day of post,
#   author account age,
#   author followers count,
#   author following count,
#   author tweet count,
#   author in list count,
#   author verified status
calc_detail_vector_size = lambda x : sum(range(x+1))
DETAIL_FEATURES = calc_detail_vector_size(8)

#create token sequences for our inputs, and get info for generating the lstm branch
def tokenize_data(data : list) -> Tuple[list, int, int, Tokenizer]:
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

    return (data, len(words), input_size, dims, tokenizer)

#adds layers as described in rt wars paper
def pre_joint_embed_layers(inputs, fc1,fc2):
    pre_joint = Dense(fc1)(inputs)
    pre_joint = ReLU()(pre_joint)
    pre_joint = Dense(fc2)(pre_joint)
    pre_joint = BatchNormalization()(pre_joint)
    pre_joint = Lambda(lambda x: tf.keras.backend.l2_normalize(x,axis=1))(pre_joint)
    
    return pre_joint

def lstm_branch(word_num, text_input, text_dimensions):
    t_branch = Embedding(word_num, text_dimensions)(text_input)
    t_branch = LSTM(256, dropout=0.3, kernel_regularizer=regularizers.l2(0.05))(t_branch)
    t_branch = pre_joint_embed_layers(t_branch,512,256)

    #t_branch.summary()
    print("Generated lstm branch")
    
    return t_branch

def cnn_branch():
    inceptionresnet = tf.keras.applications.InceptionResNetV2()
    
    for layer in inceptionresnet.layers:
        layer.trainable = False

    pre_joint = pre_joint_embed_layers(inceptionresnet.output,768,256)

    i_branch = Flatten()(pre_joint)

    i_branch = tf.keras.Model(inputs=inceptionresnet.input, outputs=i_branch)
    #i_branch.summary()
    print("Generated cnn branch")
    return (i_branch, inceptionresnet.input)

def build_model(data : str, word_count, text_input_length, text_dimensions):
    text_input = Input(shape=(text_input_length,))

    #generate branches
    (i_branch, image_input) = cnn_branch()
    t_branch = lstm_branch(word_count, text_input, text_dimensions)
    d_branch = Input(shape=(DETAIL_FEATURES,))
    #joint embedding and convolution
    joint = concatenate([i_branch.output, t_branch])
    joint = Dense(256, kernel_regularizer=regularizers.l2(0.05))(joint)
    joint = Dense(128, kernel_regularizer=regularizers.l2(0.05))(joint)
    
    #add details input
    joint = concatenate([joint, d_branch])
    model = Dense(128, kernel_regularizer=regularizers.l2(0.05))(joint)
    
    rnn_in = Input(shape=(None,1))
    rnn = SimpleRNN(128, kernel_regularizer=regularizers.l2(0.05), return_sequences=True, return_state=True)
    dist = TimeDistributed(Dense(1, activation='relu'))

    model, _ = rnn(rnn_in, initial_state=model)
    model = dist(model)

    final = tf.keras.Model(inputs = [image_input, text_input, d_branch, rnn_in], outputs=model)
    print("Generated model")
    return final

#turns a tweet into an input. tokenizer is optional, in case data is already tokenized.
def tweet_to_training_pair(tweet, image_directory, input_size, tokenizer=None):
    #input is (([image],[text],[user data]), result)

    #check for image, else produce blank image
    image = np.zeros((299,299,3))
    path = '{}/{}.jpg'.format(image_directory, tweet['id'])
    if os.path.isfile(path):
        image = np.asarray(Image.open(path))

    #get tokenized text
    if 'sequence' in tweet:
        text = tweet['text']
    else:
        if tokenizer is None:
            print('Please supply tokenizer for non-sequenced tweets')
            raise Exception
        tokens = np.array(
            tokenizer.texts_to_sequences([tweet['text']])[0]
        )
        #we pad the left side like this because we're iterating on each json object
        text = np.zeros(input_size)
        text[-len(tokens):] = tokens
    
    text = np.array(text)


    #get user data
    posted = datetime.fromtimestamp(tweet['created_at'])
    user_data = list(map(int, [
        posted.weekday(),
        (posted - posted.replace(hour=0,minute=0,second=0)).seconds,
        (datetime.now() - datetime.strptime(tweet['author_data']['created_at'], '%Y-%m-%dT%H:%M:%S.000Z')).days / 30,
        tweet['author_data']['public_metrics']['followers_count'],
        tweet['author_data']['public_metrics']['following_count'],
        tweet['author_data']['public_metrics']['tweet_count'],
        tweet['author_data']['public_metrics']['listed_count'],
        tweet['author_data']['verified']
    ]))

    products = []
    for x in range(len(user_data)-1):
        for y in range(x+1, len(user_data)):
            products.append(user_data[x] * user_data[y])

    user_features = np.array(user_data+products, dtype='float32')

    #get ground truth
    truth = int(tweet['final_metrics']['retweet_count'])

    return ((image, text, user_features), truth)

def generate_training_data(data, image_directory,text_input_size,tokenizer=None):
    i_data, t_data, u_data, truth = [], [], [], []

    for tweet in data:
        ((i,t,u),tr) = tweet_to_training_pair(tweet,image_directory,text_input_size,tokenizer)
        i_data.append(i)
        t_data.append(t)
        u_data.append(u)
        truth.append(tr)

    

    return (
        (
            np.array(i_data),
            np.array(t_data),
            np.array(u_data)
        ),
        np.array(truth).astype('float32')
    )

def main():
    args = vars(parser.parse_args())
    
    if args['dataset'][-5:] != '.json' or not os.path.isfile(args['dataset']):
        print("Please enter a path to a valid json file")
        return

    #get tokenized data and tokenizer.
    (data, word_count, text_input_length, dims, tokenizer) = tokenize_data(json.load(open(args['dataset'], 'r')))

    #load model, or create one if no directory supplied
    if args['model'] is None:
        #create and save model
        model : tf.keras.Model = build_model(data, word_count, text_input_length, dims) 
        '''
        print('Enter a name for this model: ')
        name = input()
        model._name = name
        model.save('./Models/{}'.format(name))
        print('Saved model to ./Models{}'.format(name))
        '''
    else:
        print('Loading model')
        model : tf.keras.Model = tf.keras.models.load_model(args['model'])
        print('Model loaded')
    
    plot_model(model, to_file='model_plot.png', show_shapes=True)

    #some testing code + example singular input
    data = json.load(open('./Data Sets/test.json', 'r'))
    ((i,t,u),tr) = generate_training_data(data, '.', text_input_length, tokenizer)

    model.compile()

    for x in [i,t,u]:
        print(len(x[0]))
    
    print(model.predict([i,t,u,np.array([0])]))

main()
