
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import tensorflow as tf
from config import *
input_shape = Config.INPUT_SHAPE
def Model():
    inputs = layers.Input(shape=input_shape)
    net = layers.Conv2D(32, 3, padding='same')(inputs)
    net = layers.Activation('relu')(net)

    net = layers.Conv2D(32, 3, padding='same', kernel_regularizer=regularizers.l1(0.01))(net)
    net = layers.BatchNormalization()(net)
    net = layers.Activation('relu')(net)
    net = layers.MaxPool2D((2, 2))(net)

    net = layers.Conv2D(64, 3, padding='same', kernel_regularizer=regularizers.l1(0.01))(net)
    net = layers.BatchNormalization()(net)
    net = layers.Activation('relu')(net)

    net = layers.Conv2D(64, 3, padding='same', kernel_regularizer=regularizers.l1(0.01))(net)
    net = layers.BatchNormalization()(net)
    net = layers.Activation('relu')(net)
    net = layers.MaxPool2D((2, 2))(net)

    net = layers.Flatten()(net)
    net = layers.Dense(256)(net)
    net = layers.BatchNormalization()(net)
    net = layers.Activation('relu')(net)
    net = layers.Dropout(0.5)(net)
    net = layers.Dense(5)(net)
    net = layers.Activation('sigmoid')(net)

    model = tf.keras.Model(inputs=inputs, outputs=net, name='cnn')
    print(model.summary())
    return model


def model_vgg16():
    vgg16_model = VGG16(pooling='avg', weights='imagenet', include_top=False, input_shape=(Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3))
    for layers in vgg16_model.layers:
        layers.trainable=False
    last_output = vgg16_model.layers[-1].output
    vgg_x = layers.Flatten()(last_output)
    vgg_x = layers.Dense(4096, activation='relu')(vgg_x)
    vgg_x = layers.Dropout(0.5)(vgg_x)
    vgg_x = layers.Dense(2048, activation='relu')(vgg_x)
    vgg_x = layers.Dense(1024, activation='relu')(vgg_x)
    vgg_x = layers.Dense(6, activation='softmax')(vgg_x)
    vgg16_final_model = Model(vgg16_model.input, vgg_x)

    return vgg16_final_model