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

def evaluate_model(datapath, imagepath, tokenpath, modelpath):
    by_category = {k:0 for k in range(CATEGORIES)}
    total_bc =  {k:0 for k in range(CATEGORIES)}
    print("Loading testing data")
    data = json.load(open(datapath,'r'))

    print("Loading tokenizer")
    (tokenizer,_,_,_) = pickle.load(open(tokenpath,'rb'))

    print("Loading model")
    model : tf.keras.Model = tf.keras.models.load_model(modelpath)
    
    formatted_data = generate_training_data(data, imagepath,LSTM_LENGTH,tokenizer)

    #predictions = model.predict([formatted_data.all()[1],formatted_data.all()[2]])
    predictions = model.predict(formatted_data.all()[1])
    truth       = formatted_data.truth()

    #make matrix: row for truth, column for prediction
    matrix = np.zeros(shape=(CATEGORIES,CATEGORIES))
    hits, misses = 0,0
    for (p,t) in zip(predictions,truth):
        matrix[np.argmax(t), np.argmax(p)] += 1
        total_bc[np.argmax(t)] += 1

        if np.argmax(t) == np.argmax(p):
            hits +=1
            by_category[np.argmax(t)] += 1
        else:
            misses +=1

    return (matrix, hits, misses,by_category,total_bc)

def main():
    args = vars(parser.parse_args())

    (matrix,hits,misses,by_category,total_bc) = evaluate_model(args['dataset'], args['imageset'], args['tokenizer'], args['model'])

    print('{} hits, {} misses, hit rate of {}%.'.format(hits,misses, hits/misses*100))
    print('Individual hit rate:')

    for i in range(CATEGORIES):
        if total_bc[i] != 0:
            hitrate = by_category[i]/total_bc[i]*100
        else:
            hitrate = 100
        print('\tCategory {}: {}/{}, {}%'.format(i, by_category[i],total_bc[i],hitrate))

    print(matrix)

main()

