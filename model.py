import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
from tensorflow.python.keras.layers.advanced_activations import ReLU
from tensorflow.python.keras.layers.normalization.batch_normalization import BatchNormalization


class DepthwiseSeperableConv2D(tfk.Model):
    def __init__(self, filters, kernel_size, strides=(1,1), use_bn=True, activation=ReLU):
        super().__init__()
        self.filters = filters
        self.use_bn = use_bn
        self.conv1 = tfkl.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, use_bias=False)
        self.bn1 = BatchNormalization()
        self.act1 = activation()
        self.conv2 = tfkl.Conv2D(filters=filters, kernel_size=(1, 1), use_bias=False)
        self.bn2 = BatchNormalization()
        self.act2 = activation()

    def call(self, x):
        y = self.conv1(x)
        if self.use_bn:
            y = self.bn1(y)
        y = self.act1(y)
        y = self.conv2(y)
        if self.use_bn:
            y = self.bn2(y)
        y = self.act2(y)
        return y


class MNISTModel(tfk.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = DepthwiseSeperableConv2D(64, 3)
        self.conv2 = DepthwiseSeperableConv2D(256, 3)
        self.conv3 = DepthwiseSeperableConv2D(64, 3)

        self.flatten = tfkl.Flatten()
        self.dense1 = tfkl.Dense(256, activation="relu")
        self.bn = BatchNormalization()
        self.dense2 = tfkl.Dense(10, activation="softmax")

    def call(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)

        y = self.flatten(y)
        y = self.dense1(y)
        y = self.bn(y)
        y = self.dense2(y)
        return y

class CIFARModel(tfk.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = DepthwiseSeperableConv2D(256, 3)
        self.conv2 = DepthwiseSeperableConv2D(512, 3)
        self.conv3 = DepthwiseSeperableConv2D(256, 3)
        # self.conv4 = DepthwiseSeperableConv2D(128, 3)

        self.flatten = tfkl.Flatten()
        self.dense1 = tfkl.Dense(1024, activation="relu")
        # self.bn = BatchNormalization()
        self.dense2 = tfkl.Dense(10, activation="softmax")

    def call(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        # y = self.conv4(y)

        y = self.flatten(y)
        y = self.dense1(y)
        # y = self.bn(y)
        y = self.dense2(y)
        return y


if __name__ == "__main__":
    import numpy as np
    model = MNISTModel()
    model(np.zeros([1, 28, 28, 1]))
    model.summary()
