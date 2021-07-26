
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
    net = layers.MaxPool2D((2, 2))(net)

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
    outputs = layers.Activation('sigmoid')(net)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='cnn')
    print(model.summary())
    return model

def Model2():
    inputs = layers.Input(shape=Config.INPUT_SHAPE, name='img')
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)

    x = layers.Conv2D(64, (3, 3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)
    block_1_output = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding='same')(block_1_output)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    block_2_output = layers.add([x, block_1_output])

    x = layers.Conv2D(64, (3, 3),  padding='same')(block_2_output)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)
    block_3_output = layers.add([x, block_2_output])

    x = layers.Conv2D(64, (3, 3))(block_3_output)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(5, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='model2')

    return model

def Model3():
    inputs = layers.Input(shape=input_shape)
    net = layers.Conv2D(32, 3, padding='same')(inputs)
    net = layers.Activation('relu')(net)

    net = layers.Conv2D(32, 3, padding='same')(net)
    net = layers.BatchNormalization()(net)
    net = layers.Activation('relu')(net)
    net = layers.Dropout(0.1)(net)
    net = layers.MaxPool2D((2, 2))(net)

    net = layers.Conv2D(64, 3, padding='same')(net)
    net = layers.BatchNormalization()(net)
    net = layers.Activation('relu')(net)

    net = layers.Conv2D(64, 3, padding='same')(net)
    net = layers.BatchNormalization()(net)
    net = layers.Activation('relu')(net)
    net = layers.Dropout(0.2)(net)

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

def Model_skip():
    inputs = layers.Input(shape=input_shape)
    x1 = CBR2D(inputs, 32, 3)
    x2 = CBR2D(x1, 32, 3)
    x3 = CBR2D(x2, 32, 3)
    x4 = CBR2D(x3, 32, 3)
    x5_add = layers.add([x1, x4])
    x5 = CBR2D(x5_add, 32, 3)
    x6_add = layers.add([x2, x5])
    x6 = CBR2D(x6_add, 32, 3)
    x7_add = layers.add([x3, x6])
    x7 = CBR2D(x7_add, 32, 3)
    x8_add = layers.add([x4, x7])
    x8 = CBR2D(x8_add, 32, 3)
    x8 = layers.MaxPool2D((2, 2))(x8)

    y1 = CBR2D(x8, 64, 3)
    y2 = CBR2D(y1, 64, 3)
    y3 = CBR2D(y2, 64, 3)
    y4 = CBR2D(y3, 64, 3)
    y5_add = layers.add([y1, y4])
    y5 = CBR2D(y5_add, 64, 3)
    y6_add = layers.add([y2, y5])
    y6 = CBR2D(y6_add, 64, 3)
    y7_add = layers.add([y3, y6])
    y7 = CBR2D(y7_add, 64, 3)
    y8_add = layers.add([y4, y7])
    y8 = CBR2D(y8_add, 64, 3)
    y8 = layers.MaxPool2D((2, 2))(y8)

    z1 = CBR2D(y8, 128, 3)
    z2 = CBR2D(z1, 128, 3)
    z3 = CBR2D(z2, 128, 3)
    z4 = CBR2D(z3, 128, 3)

    outputs = layers.GlobalAveragePooling2D()(z4)
    outputs = layers.Dense(5)(outputs)
    outputs = layers.Activation('sigmoid')(outputs)
    model = tf.keras.Model(inputs, outputs)
    print(model.summary())
    return model


def Model4():
    inputs = layers.Input(shape=input_shape)
    net = CBR2D(inputs, 32, 3)
    for i in range(7):
        net = CBR2D(net, 32, 3)
    net = layers.MaxPool2D((2, 2))(net)

    for i in range(8):
        net = CBR2D(net, 64, 3)
    net = layers.MaxPool2D((2, 2))(net)

    for i in range(4):
        net = CBR2D(net, 128, 3)
    net = layers.MaxPool2D((2, 2))(net)

    net = layers.GlobalAveragePooling2D()(net)
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

def Model5():
    inputs = layers.Input(shape=input_shape)
    x0 = layers.Conv2D(16, 3, padding='same')(inputs)
    x0 = layers.Activation('relu')(x0)

    x0 = layers.Conv2D(16, 3, padding='same')(x0)
    x0 = layers.Activation('relu')(x0)
    x0 = layers.MaxPool2D((2, 2))(x0)


    #inputs = layers.Input(shape=input_shape)
    x1 = layers.Conv2D(32, 3, padding='same')(x0)
    x1 = layers.Activation('relu')(x1)

    x2 = layers.Conv2D(32, 3, padding='same')(x1)
    x2 = layers.Activation('relu')(x2)

    x3 = layers.Conv2D(32, 3, padding='same')(x2)
    x3 = layers.Activation('relu')(x3)
    #x4 = layers.add([x1, x3])

    x4 = layers.Conv2D(32, 3, padding='same')(x3)
    x4 = layers.Activation('relu')(x4)
    x4 = layers.MaxPool2D((2, 2))(x4)

    y1 = layers.Conv2D(64, 3, padding='same')(x4)
    y1 = layers.BatchNormalization()(y1)
    y1 = layers.Activation('relu')(y1)

    y2 = layers.Conv2D(64, 3, padding='same')(y1)
    y2 = layers.BatchNormalization()(y2)
    y2 = layers.Activation('relu')(y2)

    y3 = layers.Conv2D(64, 3, padding='same')(y2)
    y3 = layers.BatchNormalization()(y3)
    y3 = layers.Activation('relu')(y3)
    #y4 = layers.add([y1, y3])

    y4 = layers.Conv2D(64, 3, padding='same')(y3)
    y4 = layers.BatchNormalization()(y4)
    y4 = layers.Activation('relu')(y4)
    y4 = layers.MaxPool2D((2, 2))(y4)

    z1 = layers.Conv2D(128, 3, padding='same')(y4)
    z1 = layers.BatchNormalization()(z1)
    z1 = layers.Activation('relu')(z1)
    #z1 = layers.Dropout(0.1)(z1)

    z2 = layers.Conv2D(128, 3, padding='same')(z1)
    z2 = layers.BatchNormalization()(z2)
    z2 = layers.Activation('relu')(z2)
    #z2 = layers.Dropout(0.1)(z2)

    z3 = layers.Conv2D(128, 3, padding='same')(z2)
    z3 = layers.BatchNormalization()(z3)
    z3 = layers.Activation('relu')(z3)
    #z3 = layers.Dropout(0.1)(z3)
    #z4 = layers.add([z1, z3])

    z4 = layers.Conv2D(128, 3, padding='same')(z3)
    z4 = layers.BatchNormalization()(z4)
    z4 = layers.Activation('relu')(z4)
    #z4 = layers.Dropout(0.2)(z4)
    z4 = layers.MaxPool2D((2, 2))(z4)

    z4 = layers.Conv2D(256, 3, padding='same')(z4)
    z4 = layers.BatchNormalization()(z4)
    z4 = layers.Activation('relu')(z4)

    z4 = layers.Conv2D(256, 3, padding='same')(z4)
    z4 = layers.BatchNormalization()(z4)
    z4 = layers.Activation('relu')(z4)
    #z4 = layers.Dropout(0.4)(z4)
    z4 = layers.MaxPool2D((2, 2))(z4)

    z4 = layers.GlobalAveragePooling2D()(z4)
    outputs = layers.Dense(5)(z4)
    outputs = layers.Activation('sigmoid')(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='cnn')
    print(model.summary())
    return model


def Model6():
    #inputs = layers.Input(shape=input_shape)

    inputs = layers.Input(shape=input_shape)

    x = CBR2D(inputs, 32, 3)
    x = CBR2D(x, 32, 3)
    x = layers.MaxPool2D((2, 2))(x)

    for i in range(8):
        x = CBR2D(x, 64, 3)
    x = layers.MaxPool2D((2, 2))(x)

    for i in range(8):
        x = CBR2D(x, 128, 3, 0.1)
    x = layers.MaxPool2D((2, 2))(x)




    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(5)(x)
    outputs = layers.Activation('sigmoid')(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='cnn')
    print(model.summary())
    return model

def CBR2D(net, nker, k, drop=None):
    net = layers.Conv2D(nker, k, padding='same')(net)
    net = layers.BatchNormalization()(net)
    net = layers.Activation('relu')(net)
    if drop is not None:
        net = layers.Dropout(drop)(net)
    return net

