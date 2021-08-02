
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import tensorflow as tf
from config import *
input_shape = Config.INPUT_SHAPE

def Model6():
    inputs = layers.Input(shape=input_shape)
    x = CBR2D(inputs, 64, 3)
    x = CBR2D(x, 64, 3)
    #x = CBR2D(x, 64, 3)
    x = layers.MaxPool2D((2, 2))(x)
    #x = CBR2D(x, 64, 3)

    for i in range(2):
        x = CBR2D(x, 128, 3, 0.1)
    x = layers.MaxPool2D((2, 2))(x)

    for i in range(2):
        x = CBR2D(x, 256, 3, 0.1)
    x = layers.MaxPool2D((2, 2))(x)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(5)(x)
    outputs = layers.Activation('sigmoid')(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='cnn')
    print(model.summary())
    return model


def Model_g():
    inputs = layers.Input(shape=input_shape)
    x = CBR2D(inputs, 32, 3)
    x = CBR2D(x, 32, 3)
    x = layers.MaxPool2D((2, 2))(x)
    for i in range(8):
        x = CBR2D(x, 64, 3, 0.1)
    x = layers.MaxPool2D((2, 2))(x)
    for i in range(8):
        x = CBR2D(x, 128, 3, 0.1)
    outputs = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(5)(outputs)
    outputs = layers.Activation('sigmoid')(outputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs,name='cnn')
    print(model.summary())
    return model

def Model_flat():
    inputs = layers.Input(shape=input_shape)
    x = CBR2D(inputs, 32, 3)
    x = CBR2D(x, 32, 3)
    x = CBR2D(x, 32, 3)
    x = CBR2D(x, 32, 3)
    x = layers.MaxPool2D((2, 2))(x)
    x = CBR2D(x, 64, 3)
    x = CBR2D(x, 64, 3)
    x = CBR2D(x, 64, 3)
    x = CBR2D(x, 64, 3)
    x = layers.MaxPool2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256)(x)
    x = layers.Activation('relu')(x)
    x = layers.Dense(5)(x)
    outputs = layers.Activation('sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='flat')
    print(model.summary())
    return model


def CBR2D(net, nker, k, drop=None):
    net = layers.Conv2D(nker, (k, k), padding='same')(net)
    net = layers.BatchNormalization()(net)
    net = layers.Activation('relu')(net)
    if drop is not None:
        net = layers.Dropout(drop)(net)
    return net

