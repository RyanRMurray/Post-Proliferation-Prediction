import json
from data_generator import  generate_training_data, CATEGORIES
import pickle
import tensorflow as tf
import numpy as np

import argparse
parser = argparse.ArgumentParser(description='Generate stats for models')
parser.add_argument('model', metavar='model', type=str, help='Path to the training data set')
parser.add_argument('dataset', metavar='dataset', type=str, help='Path to the data set')
parser.add_argument('tokenizer', metavar='tokenizer', type=str, help='Path to a tokenizer with associated metrics')

def evaluate_model(datapath, tokenpath, modelpath):
    by_category = {k:0 for k in range(CATEGORIES)}
    total_bc =  {k:0 for k in range(CATEGORIES)}
    print("Loading testing data")
    data = json.load(open(datapath,'r'))

    print("Loading tokenizer")
    (tokenizer,_,_,_) = pickle.load(open(tokenpath,'rb'))

    print("Loading model")
    model : tf.keras.Model = tf.keras.models.load_model(modelpath)
    
    formatted_data = generate_training_data(data,tokenizer)

    predictions = model.predict(formatted_data.all())
    truth       = formatted_data.truth()

    #make matrix: row for truth, column for prediction
    matrix = np.zeros(shape=(CATEGORIES,CATEGORIES))
    hits, misses = 0,0
    mse = 0
    for (p,t) in zip(predictions,truth):
        matrix[np.argmax(t), np.argmax(p)] += 1
        total_bc[np.argmax(t)] += 1

        if np.argmax(t) == np.argmax(p):
            hits +=1
            by_category[np.argmax(t)] += 1
        else:
            misses +=1
        
        mse += (np.argmax(t)- np.argmax(p))**2

    mse /= hits + misses

    return (matrix, hits, misses,by_category,total_bc, mse)

def main():
    args = vars(parser.parse_args())

    (matrix,hits,misses,by_category,total_bc, mse) = evaluate_model(args['dataset'], args['tokenizer'], args['model'])

    print('{} hits, {} misses, hit rate of {}%.'.format(hits,misses, (hits/(hits+misses))*100))
    print('Individual hit rate:')

    for i in range(CATEGORIES):
        if total_bc[i] != 0:
            hitrate = by_category[i]/total_bc[i]*100
        else:
            hitrate = 100
        print('\tCategory {}: {}/{}, {}%'.format(i, by_category[i],total_bc[i],hitrate))

    print(matrix)

    print("MSE = {}".format(mse))

main()

