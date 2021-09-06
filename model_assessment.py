from data_generator import CATEGORIES
from prediction_network import ModelType
import pickle
import tensorflow as tf
import numpy as np

import argparse

parser = argparse.ArgumentParser(description="Generate stats for models")
parser.add_argument(
    "model", metavar="model", type=str, help="Path to the training data set"
)
parser.add_argument("dataset", metavar="dataset", type=str, help="Path to the data set")
parser.add_argument(
    "modeltype",
    metavar="modeltype",
    type=ModelType,
    choices=list(ModelType),
    help="Version of model to assess",
)
parser.add_argument(
    "full_set",
    metavar="full_set",
    type=bool,
    default=True,
    nargs="?",
    help="Whether to use the full data set for assessment",
)


def evaluate_model(datapath: str, modelpath: str, modeltype: ModelType, full: bool):
    by_category = {k: 0 for k in range(CATEGORIES)}
    total_bc = {k: 0 for k in range(CATEGORIES)}

    print("Loading model")
    model: tf.keras.Model = tf.keras.models.load_model(modelpath)

    formatted_data = pickle.load(open(datapath, "rb"))

    if full:
        d = formatted_data.all()
        truth = formatted_data.truth()
    else:
        d = formatted_data.x_valid()
        truth = formatted_data.y_valid()

    if modeltype == ModelType.Details:
        to_eval = d[1]
    elif modeltype == ModelType.Text:
        to_eval = d[0]
    else:
        to_eval = d

    predictions = model.predict(to_eval)

    # make matrix: row for truth, column for prediction
    matrix = np.zeros(shape=(CATEGORIES, CATEGORIES))
    hits, misses = 0, 0
    mse = 0
    for (p, t) in zip(predictions, truth):
        matrix[np.argmax(t), np.argmax(p)] += 1
        total_bc[np.argmax(t)] += 1

        if np.argmax(t) == np.argmax(p):
            hits += 1
            by_category[np.argmax(t)] += 1
        else:
            misses += 1

        mse += (np.argmax(t) - np.argmax(p)) ** 2

    mse /= hits + misses

    return (matrix, hits, misses, by_category, total_bc, mse)


def main():
    args = vars(parser.parse_args())

    (matrix, hits, misses, by_category, total_bc, mse) = evaluate_model(
        args["dataset"], args["model"], args["modeltype"], args["full_set"]
    )

    print(
        "{} hits, {} misses, hit rate of {}%.".format(
            hits, misses, (hits / (hits + misses)) * 100
        )
    )
    print("Individual hit rate:")

    for i in range(CATEGORIES):
        if total_bc[i] != 0:
            hitrate = by_category[i] / total_bc[i] * 100
        else:
            hitrate = 100
        print(
            "\tCategory {}: {}/{}, {}%".format(i, by_category[i], total_bc[i], hitrate)
        )

    print(matrix)

    print("MSE = {}".format(mse))


main()
