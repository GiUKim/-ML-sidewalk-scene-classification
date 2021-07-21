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
from tensorflow.python.keras import layers
from tensorflow.python.keras import regularizers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications.vgg16 import preprocess_input
from config import Config
from tensorflow.python.keras.metrics import Recall
import tensorflow.python.keras.backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# 224 224 3
def model_vgg16():
    vgg16_model = VGG16(pooling='avg', weights='imagenet', include_top=False, input_shape=(Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3))
    for layers in vgg16_model.layers:
        layers.trainable=False
    last_output = vgg16_model.layers[-1].output
    vgg_x = Flatten()(last_output)
    vgg_x = Dense(4096, activation='relu')(vgg_x)
    vgg_x = Dropout(0.5)(vgg_x)
    vgg_x = Dense(2048, activation='relu')(vgg_x)
    vgg_x = Dense(1024, activation='relu')(vgg_x)
    vgg_x = Dense(6, activation='softmax')(vgg_x)
    vgg16_final_model = Model(vgg16_model.input, vgg_x)
    vgg16_final_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', f1_m, precision_m, recall_m])
    return vgg16_final_model

# 32 32 3
def model_my1():
    # inputs = Input(shape=(Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3))
    # x1 = Conv2D(32, (3, 3), padding='same')(inputs)
    # x1 = Activation('relu')(x1)
    # x1 = MaxPooling2D(pool_size=(2, 2))(x1)
    #
    # x2 = Conv2D(64, (3, 3), padding='same')(x1)
    # x2 = Activation('relu')(x2)
    # x2 = MaxPooling2D(pool_size=(2, 2))(x2)
    #
    # outputs = GlobalAveragePooling2D()(x2)
    # outputs = Dense(256)(outputs)
    # outputs = Dense(6, activation='softmax')(outputs)
    #
    # model = Model(inputs, outputs, name='my_model1')
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # return model

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same'
                     , input_shape=(32, 32, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256))
    #model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    model.add(Dense(5, activation='softmax'))
    #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[Recall(name='recall')])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', f1_m, precision_m, recall_m])
    return model

# 32 32 3
def model_my2():
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
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', f1_m, precision_m, recall_m])
    return model

# 64 64 3
def model_my3():
    inputs = Input(shape=(64, 64, 3), name='my3')
    x1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D(pool_size=(2, 2))(x1)

    y1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    y1 = BatchNormalization()(y1)
    y1 = MaxPooling2D(pool_size=(2, 2))(y1)

    x2_pure = Conv2D(32, (3, 3), activation='relu', padding='same')(x1)
    x2_pure = BatchNormalization()(x2_pure)
    x2 = add([x2_pure, y1])

    y2_pure = Conv2D(16, (1, 1), activation='relu', padding='same')(y1)
    y2_pure = BatchNormalization()(y2_pure)
    y2 = add([y2_pure, x1])

    x3 = Conv2D(64, (3, 3), activation='relu')(x2)
    x3 = BatchNormalization()(x3)

    x3 = Conv2D(64, (3, 3), activation='relu')(x3)
    x3 = BatchNormalization()(x3)
    x4_pure = MaxPooling2D(pool_size=(2, 2))(x3) # 14 14 64

    y3_pure = Conv2D(32, (3, 3), activation='relu', padding='same')(y2)
    y3_pure = BatchNormalization()(y3_pure)
    y3 = add([y3_pure, x2])

    y3 = Conv2D(64, (3, 3), activation='relu', padding='same')(y3)
    y3 = BatchNormalization()(y3)
    y3 = MaxPooling2D(pool_size=(2, 2))(y3)

    y4 = Conv2D(64, (3, 3), activation='relu')(y3)
    y4 = BatchNormalization()(y4)

    x4 = add([x4_pure, y4])
    x4 = MaxPooling2D(pool_size=(2, 2))(x4)

    x4 = GlobalAveragePooling2D()(x4)
    x4 = Dense(512)(x4)
    x4 = BatchNormalization()(x4)
    x4 = Activation('relu')(x4)
    x4 = Dropout(0.5)(x4)
    outputs = Dense(6, activation='softmax')(x4)
    model = Model(inputs, outputs, name='my3')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', f1_m, precision_m, recall_m])

    print(model.summary())
    return model

# 64 64 3
def model_my4():
    inputs = Input(shape=(64, 64, 3))
    x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x1 = BatchNormalization()(x1)
    x2_pure = Conv2D(32, (3, 3), activation='relu', padding='same')(x1)
    x2_pure = BatchNormalization()(x2_pure)
    x2 = add([x1, x2_pure])
    x3_pure = Conv2D(32, (3, 3), activation='relu', padding='same')(x2)
    x3_pure = BatchNormalization()(x3_pure)
    x3_pure2 = add([x1, x3_pure])
    x3 = add([x3_pure2, x2])
    y1 = MaxPooling2D(pool_size=(2, 2))(x3)

    y1_pure = Conv2D(64, (3, 3), activation='relu', padding='same')(y1)
    y1_pure = BatchNormalization()(y1_pure)
    y2_pure = Conv2D(64, (3, 3), activation='relu', padding='same')(y1_pure)
    y2_pure = BatchNormalization()(y2_pure)
    y2 = add([y1_pure, y2_pure])
    y3_pure = Conv2D(64, (3, 3), activation='relu', padding='same')(y2)
    y3_pure = BatchNormalization()(y3_pure)
    y3 = add([y3_pure, y2])
    y3 = MaxPooling2D(pool_size=(2, 2))(y3)

    y3 = GlobalAveragePooling2D()(y3)
    y3 = Dense(256, activation='relu')(y3)
    y3 = Dropout(0.5)(y3)

    outputs = Dense(6, activation='softmax')(y3)
    model = Model(inputs, outputs, name='myres')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', f1_m, precision_m, recall_m])
    return model
