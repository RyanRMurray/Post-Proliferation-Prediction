import json
import math
import pickle
from random import shuffle
import matplotlib.pyplot as plt
from nltk.tokenize import TweetTokenizer
import os
from numpy.lib.npyio import save

from tensorflow.keras import regularizers
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling1D, GlobalMaxPool1D
#from tensorflow.python.keras.layers.wrappers import Bidirectional
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import argparse
parser = argparse.ArgumentParser(description='Generate and train a model to predict success of Twitter posts')
parser.add_argument('dataset', metavar='dataset', type=str, help='Path to the training data set')
parser.add_argument('glove', metavar='glove', type=str, help='Path to the training GLoVE training set')
parser.add_argument('tokenizer', metavar='tokenizer', type=str, help='Path to a tokenizer with associated metrics')
parser.add_argument('model', metavar='model', type=str, nargs='?', help='Path to a pre-generated model (optional)', default=None)

import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Bidirectional, Dense, Flatten, Embedding, LSTM, BatchNormalization, Lambda, concatenate, Dropout, SpatialDropout1D, GRU, Conv1D, GlobalAveragePooling1D,GlobalMaxPool1D
from keras.utils.vis_utils import plot_model
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from data_generator import generate_training_data, TrainingData, DETAIL_FEATURES, CATEGORIES, SEQ_LENGTH

#class weights for imbalanced input
EPOCHS = 40 
WEIGHTS = {
    0:0.17489981752,
    1:4.17343286701,
    2:26.4272539289,
    3:215.372767105,
    4:3132.40686275,
    5:35500.6111111
}

def model_new(matrix):
    #text layer
    t_input = Input(shape=(SEQ_LENGTH,))
    t_branch = Embedding(100_000, 100, input_length=SEQ_LENGTH, weights=[matrix], trainable=False)(t_input)
    t_branch = SpatialDropout1D(0.3)(t_branch)

    gru_layer = Bidirectional(GRU(128,return_sequences=True))(t_branch)

    p1 = Conv1D(32, kernel_size=4, kernel_initializer='he_uniform')(gru_layer)
    gru1_avg = GlobalAveragePooling1D()(p1)
    gru1_max = GlobalMaxPool1D()(p1)

    p2 = Conv1D(32, kernel_size=2, kernel_initializer='he_uniform')(gru_layer)
    gru2_avg = GlobalAveragePooling1D()(p2)
    gru2_max = GlobalMaxPool1D()(p2)

    lstm_layer = Bidirectional(LSTM(128, return_sequences=True))(t_branch)

    p1 = Conv1D(32, kernel_size=2, kernel_initializer='he_uniform')(lstm_layer)
    lstm1_avg = GlobalAveragePooling1D()(p1)
    lstm1_max = GlobalMaxPool1D()(p1)

    p2 = Conv1D(32, kernel_size=2, kernel_initializer='he_uniform')(lstm_layer)
    lstm2_avg = GlobalAveragePooling1D()(p2)
    lstm2_max = GlobalMaxPool1D()(p2)

    t_branch = concatenate([gru1_avg, gru1_max, gru2_avg, gru2_max, lstm1_avg, lstm1_max, lstm2_avg, lstm2_max])
    t_branch = BatchNormalization()(t_branch)
    t_branch = Dense(64, activation='relu')(t_branch)
    t_branch = Dropout(0.2)(t_branch)
    t_branch = BatchNormalization()(t_branch)
    t_branch = Dense(32, activation='relu')(t_branch)
    t_branch = Dropout(0.2)(t_branch)
    #author features layer
    d_input = Input(shape=(DETAIL_FEATURES,))
    d_branch = Dense(128)(d_input)
    d_branch = Dropout(0.2)(d_branch)

    joint =  concatenate([t_branch, d_branch])
    model = Dense(128)(joint)
    #model = Dense(128)(d_branch)
    model = Dropout(0.1)(model)
    model = Dense(32)(model)
    model = Dropout(0.1)(model)

    #output layer
    model = Dense(CATEGORIES, activation='softmax')(model)

    final = tf.keras.Model(inputs=[t_input, d_input], outputs = model)
    #final = tf.keras.Model(inputs= d_input, outputs = model)
    print("Generated Model")
    final.summary()
    plot_model(final, to_file='full_model_plot.png', show_shapes=True)

    return final

def embedding_matrix(tokenizer, glovepath):
    embedding_index = {}

    with open(glovepath, 'r') as g:
        for line in g:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coefs
        
    matrix = np.zeros((100_000, 100))

    for word, i in tokenizer.word_index.items():
        if i > 100_001:
            break
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            matrix[i] = embedding_vector
    
    return matrix

def main():
    print('Enter a name for this model: ')
    name = input()
    args = vars(parser.parse_args())

    #get tokenizer
    print('Loading tokenizer')
    with open(args['tokenizer'], 'rb') as f:
        (tokenizer, word_count,text_input_length,dims) = pickle.load(f)
    
    print('Loading Training Data')
    if not os.path.isfile(args['dataset']):
        print('Please enter a path to a valid file')
        return
    else:
        if args['dataset'][-5:] == '.json':
            print('Loading data set')
            with open(args['dataset'],'r') as f:
                data = json.load(f)
                print('Loaded file.')

            print('Generating data set')
            formatted : TrainingData  = generate_training_data(data, tokenizer=tokenizer)
        elif args['dataset'][-7:] == '.pickle':
            with open(args['dataset'],'rb') as f:
                formatted : TrainingData = pickle.load(f)
        else:   
            print('Please enter a path to a valid json/pickle file')
            return

    print('Generating Embedding Matrix')
    matrix = embedding_matrix(tokenizer, args['glove'])
    
    print('Generating Model.')
    model = model_new(matrix)

    print('Training Model')
    model.compile(optimizer=Adamax(learning_rate=0.001) , metrics=['accuracy'], loss='categorical_crossentropy')

    model.save('./Models/{}'.format(name))
    
    check = ModelCheckpoint('./Models/Checkpoints/{}'.format(name), monitor='accuracy', save_best_only=True)
    logger = CSVLogger('./Models/Checkpoints/{}/history.csv'.format(name), append=True)

    h = model.fit(
        x=formatted.x_train(),
        #x=formatted.x_train()[1],
        y=formatted.y_train(),
        validation_data=(formatted.x_valid(),formatted.y_valid()),
        #validation_data=(formatted.x_valid()[1],formatted.y_valid()),
        epochs=EPOCHS,
        batch_size=500,
        verbose=1,
        class_weight=WEIGHTS,
        callbacks=[check,logger]
    )

    model.save('./Models/{}_Trained'.format(name))
    print('Saved model to ./Models/{}_Trained'.format(name))

    plt.plot(h.history['accuracy'])
    plt.plot(h.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./Images/full_training_acc.png')
    plt.clf()
    
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./Images/full_training_loss.png')

main()

#Code beneath this line is depricated but kept for completeness
###############################################################################

LSTM_GENERATIONS = 100
LSTM_LENGTH      = 150
IMAGE_SHAPE = (1)
DEFAULT_IMAGE = np.zeros(IMAGE_SHAPE)

def lstm_branch(name, data, word_num, text_input, text_dimensions):
    directory_path = './Models/{}/LSTM'.format(name)

    if os.path.isdir(directory_path):
        print('Loading trained LSTM Branch')
        t_branch = tf.keras.models.load_model(directory_path)
        print('Loaded')
    else:
        print('Generating LSTM Branch')
        #t_branch = Embedding(word_num, text_dimensions)(text_input)
        t_branch = Embedding(100_000, 50, input_length=LSTM_LENGTH)(text_input)
        t_branch = LSTM(256, kernel_regularizer=regularizers.l2(0.05))(t_branch)

        #train t_branch
        t_branch = Dense(CATEGORIES, activation='softmax')(t_branch)
        
        t_branch = tf.keras.Model(inputs=text_input, outputs= t_branch)
        '''
        plot_model(t_branch, to_file='./Images/lstm_plot.png', show_shapes=True)

        t_branch.summary()
        t_branch.compile(optimizer='Adam', metrics=['accuracy'], loss='categorical_crossentropy')
        h = t_branch.fit(
            x=data.x_train()[1],
            y=data.y_train(),
            validation_data=(data.x_valid()[1],data.y_valid()),
            epochs=LSTM_GENERATIONS,
            batch_size=1000,
            verbose=1,
            class_weight=WEIGHTS
        )

        plt.plot(h.history['accuracy'])
        plt.plot(h.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('./Images/lstm_training.png')
        print('Trained LSTM branch. Saving...')
        t_branch.save(directory_path)

        t_branch = tf.keras.models.load_model(directory_path)

    for l in t_branch.layers:
        l.trainable = False
    '''
    t_branch = tf.keras.Model(inputs=t_branch.input , outputs= t_branch.layers[-2].output)
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

def build_model_test(name, data, word_count, text_input_length, text_dimensions):
    text_input = Input(shape=(LSTM_LENGTH,))

    #generate branches
    t_input = lstm_branch(name, data, word_count, text_input, text_dimensions)
    t_branch = BatchNormalization()(t_input.output)
    d_branch = Dropout(0.2)(t_branch)

    d_input = Input(shape=(DETAIL_FEATURES,))
    d_branch = Dense(256)(d_input)
    d_branch = Dropout(0.2)(d_branch)
    joint = concatenate([t_branch, d_branch])
    model = Dense(256, kernel_regularizer=regularizers.l2(0.05))(joint)
    model = Dropout(0.4)(model)
    model = Dense(128, kernel_regularizer=regularizers.l2(0.05))(model)
    model = Dropout(0.4)(model)
    
    #output layer
    model = Dense(CATEGORIES, activation='softmax')(model)
    
    final = tf.keras.Model(inputs=[t_input.input, d_input], outputs = model)
    print("Generated Model")
    final.summary()

    return final

def build_model(name, data, word_count, text_input_length, text_dimensions):
    text_input = Input(shape=(LSTM_LENGTH,))

    #generate branches
    t_branch = lstm_branch(name, data, word_count, text_input, text_dimensions)
    
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

#create a tokenizer
def create_tokenizer_old(data, max_words=None):
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

def main1():
    print('Enter a name for this model: ')
    name = input()
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
        model : tf.keras.Model = build_model_test(name, formatted_data, word_count, text_input_length, dims)
        #model : tf.keras.Model = build_model(name, formatted_data, word_count, text_input_length, dims)
        model._name = name
        model.save('./Models/{}'.format(name))
        print('Saved model to ./Models/{}'.format(name))
    else:
        print('Loading model')
        model : tf.keras.Model = tf.keras.models.load_model(args['model'])
        print('Model loaded')
    
    plot_model(model, to_file='full_model_plot.png', show_shapes=True)

    '''
    #some testing code + example singular input
    data = json.load(open('./Data Sets/test.json', 'r'))
    ((i,t,u),tr) = generate_training_data(data, '.', text_input_length, tokenizer)

    model.compile()
    
    print(model.predict([i,t,u]))
    '''
    model.compile(optimizer=Adamax(learning_rate=0.0001, clipvalue=5) , metrics=['accuracy'], loss='categorical_crossentropy')
    model.summary()

    save_every = (len(data) // 1000) * 10 #unsimplified for clarity: save every 100 epochs
    checkpointing = ModelCheckpoint(
        filepath='Models/checkpoints/{}'.format(name),
        save_weights_only=True,
        monitor='accuracy',
        mode='max',
        save_best_only=True,
        save_freq=save_every
    )
    h = model.fit(
        x=[formatted_data.x_train()[1],formatted_data.x_train()[2]],
        y=formatted_data.y_train(),
        validation_data=([formatted_data.x_valid()[1],formatted_data.x_valid()[2]],formatted_data.y_valid()),
        epochs=500,
        #batch_size=1000,
        verbose=1,
        class_weight=WEIGHTS,
        callbacks=[checkpointing]
        #validation_steps=len(data.y_valid())//100,
        #steps_per_epoch=len(data.y_train())//100
    )

    plt.plot(h.history['accuracy'])
    plt.plot(h.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./Images/full_training_acc.png')
    plt.show()

    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./Images/full_training_loss.png')

    model.save('./Models/{}_Trained'.format(name))
    print('Saved model to ./Models/{}_Trained'.format(name))
