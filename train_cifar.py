import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
from model import CIFARModel

import numpy as np

(x_train, y_train), (x_test, y_test) = tfk.datasets.cifar10.load_data()

print(x_train.shape)
x_train = x_train.astype('float')/127.5 - 1
x_test = x_test.astype('float')/127.5 - 1

model = CIFARModel()

model.compile(optimizer=tfk.optimizers.Adam(learning_rate=1e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=100, epochs=200, validation_data=(x_test, y_test))
model.save_weights("./cifar.h5")
