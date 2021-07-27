import json

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import cv2

inceptionresnet = tf.keras.applications.InceptionResNetV2()
inception_output = inceptionresnet.layers[-2].output
i_branch = tf.keras.Model(inputs = inceptionresnet.input, outputs = inception_output)
