import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
from model import MNISTModel

import numpy as np

(x_train, y_train), (x_test, y_test) = tfk.datasets.mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1).astype('float')/127.5 - 1
x_test = x_test.reshape(-1, 28, 28, 1).astype('float')/127.5 - 1

model = MNISTModel()

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=200, epochs=10, validation_data=(x_test, y_test))
