import json
from data_generator import  generate_training_data, CATEGORIES, LSTM_LENGTH
import pickle
import tensorflow as tf
import numpy as np

import argparse
parser = argparse.ArgumentParser(description='Generate stats for models')
parser.add_argument('model', metavar='model', type=str, help='Path to the training data set')
parser.add_argument('dataset', metavar='dataset', type=str, help='Path to the data set')
parser.add_argument('tokenizer', metavar='tokenizer', type=str, help='Path to a tokenizer with associated metrics')
parser.add_argument('imageset', metavar='imageset', type=str, help='Path to the training data set\'s images')

def evaluate_model():
    args = vars(parser.parse_args())

    print("Loading testing data")
    data = json.load(open(args['dataset'],'r'))

    print("Loading tokenizer")
    (tokenizer,_,_,_) = pickle.load(open(args['dataset'],'rb'))

    print("Loading model")
    model : tf.keras.Model = tf.keras.models.load_model(args['model'])
    
    formatted_data = generate_training_data(data, args['imageset'],LSTM_LENGTH,tokenizer)

    predictions = model.predict([formatted_data.all()[1],formatted_data.all()[2]])
    truth       = formatted_data.truth()

    #make matrix: row for truth, column for prediction
    matrix = np.zeros(shape=(CATEGORIES,CATEGORIES))
    hits, misses = 0,0
    for (p,t) in zip(predictions,truth):
        matrix[np.argmax(t), np.argmax(p)] += 1

        if np.argmax(t) == np.argmax(p):
            hits +=1
        else:
            misses +=1

    return (matrix, hits, misses)

