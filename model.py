from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import add
from tensorflow.python.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras import layers
from tensorflow.python.keras import regularizers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications.vgg16 import preprocess_input
from config import Config

def model_myres2():
    inputs = Input(shape=(Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3))
    x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x2_pure = Conv2D(32, (3, 3), activation='relu', padding='same')(x1)
    x2 = add([x1, x2_pure])
    x3_pure = Conv2D(32, (3, 3), activation='relu', padding='same')(x2)
    x3_pure2 = add([x1, x3_pure])
    x3 = add([x3_pure2, x2])
    y1 = MaxPooling2D(pool_size=(2, 2))(x3)

    y1_pure = Conv2D(64, (3, 3), activation='relu', padding='same')(y1)
    y2_pure = Conv2D(64, (3, 3), activation='relu', padding='same')(y1_pure)
    y2 = add([y1_pure, y2_pure])
    y3_pure = Conv2D(64, (3, 3), activation='relu', padding='same')(y2)
    y3 = add([y3_pure, y2])
    y3 = MaxPooling2D(pool_size=(2, 2))(y3)

    y3 = GlobalAveragePooling2D()(y3)
    y3 = Dense(256, activation='relu')(y3)
    y3 = Dropout(0.5)(y3)

    outputs = Dense(6, activation='softmax')(y3)
    model = Model(inputs, outputs, name='myres')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
def model_org():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same'
                     , input_shape=(Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    model.add(Dense(6, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
def model_vgg16():
    vgg16_model = VGG16(pooling='avg', weights='imagenet', include_top=False, input_shape=(Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3))
    for layers in vgg16_model.layers:
        layers.trainable=False
    last_output = vgg16_model.layers[-1].output
    vgg_x = Flatten()(last_output)
    vgg_x = Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(0.01))(vgg_x)
    vgg_x = Dropout(0.5)(vgg_x)
    vgg_x = Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(0.01))(vgg_x)
    vgg_x = Dropout(0.1)(vgg_x)
    vgg_x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01))(vgg_x)
    vgg_x = Dropout(0.1)(vgg_x)
    vgg_x = Dense(6, activation='softmax', kernel_regularizer=regularizers.l2(0.01))(vgg_x)
    vgg16_final_model = Model(vgg16_model.input, vgg_x)
    vgg16_final_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return vgg16_final_model
def model_myres1():
    inputs = Input(shape=(32, 32, 3), name='img')
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    block_1_output = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(block_1_output)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    block_2_output = add([x, block_1_output])

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(block_2_output)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    block_3_output = add([x, block_2_output])

    x = Conv2D(64, (3, 3), activation='relu')(block_3_output)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(6, activation='softmax')(x)

    model = Model(inputs, outputs, name='small_residual')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model