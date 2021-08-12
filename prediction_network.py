import json
import cv2
import re
import math
import itertools
import sys
import pickle
from datetime import datetime
from PIL import Image
import gc
import matplotlib.pyplot as plt
from nltk.tokenize import TweetTokenizer
import os

from tensorflow.keras import regularizers
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import argparse
parser = argparse.ArgumentParser(description='Generate and train a model to predict success of Twitter posts')
parser.add_argument('dataset', metavar='dataset', type=str, help='Path to the training data set')
parser.add_argument('imageset', metavar='imageset', type=str, help='Path to the training data set\'s images')
parser.add_argument('tokenizer', metavar='tokenizer', type=str, help='Path to a tokenizer with associated metrics')
parser.add_argument('model', metavar='model', type=str, nargs='?', help='Path to a pre-generated model (optional)', default=None)

from tensorflow.python.keras.layers.recurrent import SimpleRNN
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras.utils.np_utils

from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, Flatten, Embedding, LSTM, ReLU, BatchNormalization, Lambda, Concatenate, Reshape, concatenate, SimpleRNN, TimeDistributed
from tensorflow.keras.regularizers import L2
from tensorflow.keras.models import Sequential
from keras.utils.vis_utils import plot_model

from typing import Tuple, List

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
CATEGORIES      = 7
LSTM_GENERATIONS = 1000
LSTM_LENGTH      = 150
DEFAULT_IMAGE = np.zeros((150,150,3))

class TrainingData():
    def __init__(self, i_data, t_data, u_data, truth):

        #split training/validation
        s = int(len(i_data)*0.8)
        (self.i_train,     self.i_valid)     = np.split(i_data, [s])
        (self.t_train,     self.t_valid)     = np.split(t_data, [s])
        (self.u_train,     self.u_valid)     = np.split(u_data, [s])
        (self.truth_train, self.truth_valid) = np.split(truth, [s])

    
    def x_train(self):
        return [self.i_train, self.t_train, self.u_train]
    
    def y_train(self):
        return self.truth_train

    def x_valid(self):
        return [self.i_valid, self.t_valid, self.u_valid]
    
    def y_valid(self):
        return self.truth_valid

#create a tokenizer
def create_tokenizer(data, max_words=None):
    tweets = len(data)
    words = set()
    input_size = 0
    splitter = TweetTokenizer(strip_handles=True, reduce_len=True, preserve_case=False)

    data = [splitter.tokenize(tweet) for tweet in data]

    for tweet in data:
        input_size = max(input_size, len(tweet))
        words.update(tweet)
    
    dims = math.ceil(len(words) ** (1/4))

    if max_words is None:
        print('{} unique symbols, input size is {}. Using {} dimensions.'.format(len(words), input_size, dims))
        tokenizer = Tokenizer(num_words=len(words))
    else:
        print('{} unique symbols, truncating to {}, input size is {}. Using {} dimensions.'.format(len(words), max_words, input_size, dims))
        tokenizer = Tokenizer(num_words=max_words)

    tokenizer.fit_on_texts(data)

    to_pickle = (tokenizer, len(words), input_size, dims)
    print('Enter a name for this tokenizer: ')
    name = input()
    with open('./Tokenizers/{}.pickle'.format(name), 'wb') as f:
        pickle.dump(to_pickle, f)

#adds layers as described in rt wars paper
def pre_joint_embed_layers(inputs, fc1,fc2):
    pre_joint = Dense(fc1, activation='relu')(inputs)
    pre_joint = Dense(fc2)(pre_joint)
    pre_joint = BatchNormalization()(pre_joint)
    pre_joint = Lambda(lambda x: tf.keras.backend.l2_normalize(x,axis=1))(pre_joint)
    
    return pre_joint

def lstm_branch(name, data, word_num, text_input, text_dimensions):
    directory_path = './Models/{}/LSTM'.format(name)
    #class weights for imbalanced input
    weights = {
        0:1,
        1:1,
        2:10,
        3:100,
        4:1000,
        5:10000,
        6:100000
    }

    if os.path.isdir(directory_path):
        print('Loading trained LSTM Branch')
        t_branch = tf.keras.models.load_model(directory_path)
        print('Loaded')
    else:
        print('Generating LSTM Branch')
        #t_branch = Embedding(word_num, text_dimensions)(text_input)
        t_branch = Embedding(50_000, 100)(text_input)
        t_branch = LSTM(256, dropout=0.2, kernel_regularizer=regularizers.l2(0.05))(t_branch)

        #train t_branch
        t_branch = Dense(CATEGORIES, activation='softmax')(t_branch)

        t_branch = tf.keras.Model(inputs=text_input, outputs= t_branch)
        t_branch.summary()
        plot_model(t_branch, to_file='model_plot.png', show_shapes=True)

        print(data.x_valid()[1].dtype)
        print(data.y_valid().shape)

        t_branch.compile(optimizer='Adam', metrics=['accuracy'], loss='categorical_crossentropy')
        h = t_branch.fit(
            x=data.x_train()[1],
            y=data.y_train(),
            validation_data=(data.x_valid()[1],data.y_valid()),
            epochs=5,
            #epochs=LSTM_GENERATIONS,
            #batch_size=1000,
            verbose=1,
            class_weight=weights
        )


        plt.plot(h.history['accuracy'])
        plt.plot(h.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('lstm_training.png')

        print('Trained LSTM branch. Saving...')
        t_branch.save(directory_path)

    print('Attaching branch input to LSTM layer')
    t_branch = pre_joint_embed_layers(t_branch.layers[-2].output,512,256)
    print("Generated lstm branch")
    
    return t_branch

def cnn_branch():
    inceptionresnet = tf.keras.applications.InceptionResNetV2(include_top=False, input_shape=(150,150,3))
    
    for layer in inceptionresnet.layers:
        layer.trainable = False

    #remove final softmax layer
    pre_joint = pre_joint_embed_layers(inceptionresnet.layers[-2].output,768,256)

    i_branch = Flatten()(pre_joint)

    i_branch = tf.keras.Model(inputs=inceptionresnet.input, outputs=i_branch)
    #i_branch.summary()
    print("Generated cnn branch")
    return (i_branch, inceptionresnet.input)

def build_model(name, data, word_count, text_input_length, text_dimensions):
    text_input = Input(shape=(LSTM_LENGTH,))

    #generate branches
    t_branch = lstm_branch(name, data, word_count, text_input, text_dimensions)
    raise Exception
    (i_branch, image_input) = cnn_branch()
    d_branch = Input(shape=(DETAIL_FEATURES,))
    #joint embedding and convolution
    joint = concatenate([i_branch.output, t_branch])
    joint = Dense(256, kernel_regularizer=regularizers.l2(0.05))(joint)
    joint = Dense(128, kernel_regularizer=regularizers.l2(0.05))(joint)
    
    #add details input
    joint = concatenate([joint, d_branch])
    model = Dense(128, kernel_regularizer=regularizers.l2(0.05))(joint)
    model = Dense(64, activation='relu')(model)
    
    #output layer
    model = Dense(CATEGORIES, activation='softmax')(model)

    final = tf.keras.Model(inputs = [image_input, text_input, d_branch], outputs=model)
    print("Generated model")
    return final

#turns a tweet into an input. tokenizer is optional, in case data is already tokenized.
def tweet_to_training_pair(tweet, image_directory, input_sizeone):

    '''
    #check for image, else produce blank image
    path = '{}/{}.jpg'.format(image_directory, tweet['id'])
    if os.path.isfile(path):
        image = np.asarray(Image.open(path, ).convert('RGB').resize((150,150)))
    else:
        image = DEFAULT_IMAGE
    '''

    #get text
    text = tweet['text']

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

    #get ground truth for one-hot encoding
    #one class for 0, 1 class for magnitude 1, ect.
    #we'll use 6 categories for now, with the 6th being anything above 100k RTs
    rts = int(tweet['final_metrics']['retweet_count'])
    if rts >= 100_000:
        mag = 6
    else:
        mag = 0 if rts == 0 else int(math.log10(rts))+1

    return ((None, text, user_features), mag)

def generate_training_data(data, image_directory,text_input_size,tokenizer=None):
    i_data, t_data, u_data, truth = [], [], [], []
    splitter = TweetTokenizer(strip_handles=True, reduce_len=True, preserve_case=False)
    tweets = len(data)

    counter = 0
    for tweet in data:
        ((_,t,u),tr) = tweet_to_training_pair(tweet,image_directory,text_input_size)

        #i_data.append(i)
        i_data.append([0])
        t_data.append(t)
        u_data.append(u)
        truth.append(tr)
        counter +=1 
        print('Converted {}/{} tweets'.format(counter,tweets), end='\r')

    print()
    print('Tokenizing...')
    #tokenize
    t_data = tokenizer.texts_to_sequences([splitter.tokenize(t)  if len(splitter.tokenize(t)) > 0 else [0] for t in t_data])
    t_data = pad_sequences(t_data, maxlen=LSTM_LENGTH)
    print('Done.')

    #shuffle everything in unison
    p = np.random.permutation(len(i_data))
    i_data = np.array(i_data, dtype='uint8')[p]
    t_data = np.array(t_data, dtype='uint16')[p]
    u_data = np.array(u_data)[p]
    truth  = keras.utils.np_utils.to_categorical(truth, CATEGORIES)[p]
    return TrainingData(i_data, t_data, u_data, truth)

def main():
    args = vars(parser.parse_args())

    if not os.path.isdir(args['imageset']):
        print('Please enter a path to a valid image directory')
    
    if args['dataset'][-5:] != '.json' or not os.path.isfile(args['dataset']):
        print('Please enter a path to a valid json file')
        return
    else:
        print('Loading data set')
        with open(args['dataset'],'r') as f:
            data = json.load(f)
            print('Loaded file.')

    #get tokenizer
    with open(args['tokenizer'], 'rb') as f:
        (tokenizer, word_count,text_input_length,dims) = pickle.load(f)

    print('Generating training/validation data')
    formatted_data = generate_training_data(data, args['imageset'], text_input_length, tokenizer)
    print('Generated Training data')
    
    #load model, or create one if no directory supplied
    if args['model'] is None:
        #create and save model
        print('Creating model from input.')
        print('Enter a name for this model: ')
        name = input()
        model : tf.keras.Model = build_model(name, formatted_data, word_count, text_input_length, dims)
        model._name = name
        model.save('./Models/{}'.format(name))
        print('Saved model to ./Models{}'.format(name))
    else:
        print('Loading model')
        model : tf.keras.Model = tf.keras.models.load_model(args['model'])
        print('Model loaded')
    
    plot_model(model, to_file='model_plot.png', show_shapes=True)

    '''
    #some testing code + example singular input
    data = json.load(open('./Data Sets/test.json', 'r'))
    ((i,t,u),tr) = generate_training_data(data, '.', text_input_length, tokenizer)

    model.compile()
    
    print(model.predict([i,t,u]))
    '''
    
main()
