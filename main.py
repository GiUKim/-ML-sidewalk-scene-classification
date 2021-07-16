from config import Config
import numpy as np
import os
import PIL
import PIL.Image
#import tensorflow as tf
#import tensorflow_datasets as tfds
#import pathlib
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras import layers
from tensorflow.python.keras import regularizers
import sys
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot
#from matplotlib.image import imread
import time
from tensorflow.python.keras.preprocessing.image import load_img
from tensorflow.python.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.models import load_model
import cv2
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.callbacks import ModelCheckpoint

from tensorflow.python.keras.applications.resnet import ResNet50
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.applications.mobilenet_v2 import mobilenet_v2
from tensorflow.python.keras import Model
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications.vgg16 import preprocess_input

#import autokeras as ak

# dataset/
#       upper/
#       lower/
#       one_human/
#       multi_human/
#       cycle/
#       non_human/

# Press the green button in the gutter to run the script.

def summarize_diagnostics(history):
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.legend()
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    pyplot.legend()
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.legend()
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    pyplot.legend()
    pyplot.tight_layout()
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + '_plot' + '.png')
    pyplot.close()

def model_structure():

    # vgg16_model = VGG16(pooling='avg', weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    # for layers in vgg16_model.layers:
    #     layers.trainable=False
    # last_output = vgg16_model.layers[-1].output
    # vgg_x = Flatten()(last_output)
    # vgg_x = Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(0.01))(vgg_x)
    # vgg_x = Dropout(0.5)(vgg_x)
    # vgg_x = Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(0.01))(vgg_x)
    # vgg_x = Dropout(0.1)(vgg_x)
    # vgg_x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01))(vgg_x)
    # vgg_x = Dropout(0.1)(vgg_x)
    # vgg_x = Dense(6, activation='softmax', kernel_regularizer=regularizers.l2(0.01))(vgg_x)
    # vgg16_final_model = Model(vgg16_model.input, vgg_x)
    # vgg16_final_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # pre_trained_vgg = VGG16(include_top=False, weights='imagenet', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    # pre_trained_vgg.trainable=False
    # pre_trained_vgg.summary()
    # model = Sequential()
    # model.add(pre_trained_vgg)
    # model.add(Flatten())
    # model.add(Dense(4096, activation='relu'))
    # model.add(BatchNormalization())
    # #model.add(Dropout(0.5))
    # model.add(Dense(2048, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dense(1024, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dense(6, activation='softmax'))
    # model.summary()
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

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
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    model.add(Dense(6, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
    #return classifier
    #return vgg16_final_model

def train():
    model = model_structure()
    print(model.summary())
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       validation_split=0.1
                                       )
    #test_datagen = ImageDataGenerator(rescale=1. / 255)
    #val_datagen = ImageDataGenerator(rescale=1. / 255)
    #val_datagen = ImageDataGenerator(validation_split=0.1)
    training_set = train_datagen.flow_from_directory(Config.TRAIN_DIR,
                                                     target_size=(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                                                     batch_size=Config.BATCH_SIZE,
                                                     classes=Config.CLASSES,
                                                     class_mode='categorical',
                                                     shuffle=True,
                                                     subset='training'
                                                     )
    val_set = train_datagen.flow_from_directory(Config.TRAIN_DIR,
                                                target_size=(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                                                batch_size=Config.BATCH_SIZE,
                                                classes=Config.CLASSES,
                                                class_mode='categorical',
                                                shuffle=False,
                                                subset='validation'
                                               )
    early_stopping = EarlyStopping()
    print(len(training_set))
    print(len(val_set))
    mc = ModelCheckpoint('bestmodel_' + time.strftime("%Y%m%d_%H%M%S") + '.h5', monitor='val_loss', mode='max', save_best_only=True)
    history = model.fit(training_set,
                        steps_per_epoch=len(training_set),
                        epochs=Config.EPOCHS,
                        validation_data=val_set,
                        validation_steps=len(val_set),
                        verbose=1,
                        callbacks=[mc])

    model.save('model_' + time.strftime("%Y%m%d_%H%M%S") + '.h5')
    print("-- Evaluate --")
    #_, acc = model.evaluate_generator(test_set, steps=len(test_set), verbose=0)
    _, acc = model.evaluate_generator(val_set, steps=len(val_set), verbose=0)
    print('Accuracy: {} %'.format(round(acc * 100.0, 3)))
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    print(val_set.class_indices)
    # test_set.reset()
    # print("-- Predict --")
    # output = model.predict_generator(val_set)
    # np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    # print(val_set.class_indices)
    # print(output)

    summarize_diagnostics(history)
    return round(acc * 100.0, 3)

def load_image(filename):
    img = load_img(filename, target_size=(Config.IMAGE_SIZE, Config.IMAGE_SIZE))
    img = img_to_array(img)
    img = img.reshape(1, Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3)
    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]
    return img


# batch norm 쓰고 가장 좋은거 bestmodel_20210715_165550
def run_classifier():
    model = load_model(Config.MODEL_NAME)
    # img = load_image('C:/Users/AI/PycharmProjects/class/datasets/test/upper/20210514_181704_person_2.jpg')
    # result = model.predict(img)
    # print(result[0])
    progress = 0
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # cuda open success 안뜨게하기

    predict_dataset = 'C:/Users/AI/PycharmProjects/class/dup_image_person12'
    for filename in os.listdir(predict_dataset):
        if filename.endswith(".jpg"):
            progress += 1
            print("progress: {}, filename: {} ".format(progress, filename))
            img = load_image(os.path.join(predict_dataset, filename))
            result = model.predict_generator(img)
            #print(result)

            # 헷갈려하는 사진들이 뭔지 찾기
            # if np.max(result[0]) < 0.8:
            #     org_img = cv2.imread(os.path.join(predict_dataset, filename), cv2.IMREAD_COLOR)
            #     cv2.imwrite(os.path.join('dontknow/', filename), org_img)

            if np.argmax(result[0]) == 0:
                org_img = cv2.imread(os.path.join(predict_dataset, filename), cv2.IMREAD_COLOR)
                cv2.imwrite(os.path.join('C:/Users/AI/PycharmProjects/class/new_classified/cycle/', filename), org_img)
            elif np.argmax(result[0]) == 1:
                org_img = cv2.imread(os.path.join(predict_dataset, filename), cv2.IMREAD_COLOR)
                cv2.imwrite(os.path.join('C:/Users/AI/PycharmProjects/class/new_classified/lower/', filename), org_img)
            elif np.argmax(result[0]) == 2:
                org_img = cv2.imread(os.path.join(predict_dataset, filename), cv2.IMREAD_COLOR)
                cv2.imwrite(os.path.join('C:/Users/AI/PycharmProjects/class/new_classified/multi_human/', filename), org_img)
            elif np.argmax(result[0]) == 3:
                org_img = cv2.imread(os.path.join(predict_dataset, filename), cv2.IMREAD_COLOR)
                cv2.imwrite(os.path.join('C:/Users/AI/PycharmProjects/class/new_classified/non_human/', filename), org_img)
            elif np.argmax(result[0]) == 4:
                org_img = cv2.imread(os.path.join(predict_dataset, filename), cv2.IMREAD_COLOR)
                cv2.imwrite(os.path.join('C:/Users/AI/PycharmProjects/class/new_classified/one_human/', filename), org_img)
            elif np.argmax(result[0]) == 5:
                org_img = cv2.imread(os.path.join(predict_dataset, filename), cv2.IMREAD_COLOR)
                cv2.imwrite(os.path.join('C:/Users/AI/PycharmProjects/class/new_classified/upper/', filename), org_img)

            # if np.argmax(result[0]) == 0:
            #     org_img = cv2.imread(os.path.join(predict_dataset, filename), cv2.IMREAD_COLOR)
            #     cv2.imwrite(os.path.join('C:/Users/AI/PycharmProjects/class/new_classified/cycle/', filename), org_img)
            # elif np.argmax(result[0]) == 1:
            #     org_img = cv2.imread(os.path.join(predict_dataset, filename), cv2.IMREAD_COLOR)
            #     cv2.imwrite(os.path.join('C:/Users/AI/PycharmProjects/class/new_classified/lower/', filename), org_img)
            # elif np.argmax(result[0]) == 2:
            #     org_img = cv2.imread(os.path.join(predict_dataset, filename), cv2.IMREAD_COLOR)
            #     cv2.imwrite(os.path.join('C:/Users/AI/PycharmProjects/class/new_classified/multi_human/', filename), org_img)
            #
            # elif np.argmax(result[0]) == 3:
            #     org_img = cv2.imread(os.path.join(predict_dataset, filename), cv2.IMREAD_COLOR)
            #     cv2.imwrite(os.path.join('C:/Users/AI/PycharmProjects/class/new_classified/one_human/', filename), org_img)
            # elif np.argmax(result[0]) == 4:
            #     org_img = cv2.imread(os.path.join(predict_dataset, filename), cv2.IMREAD_COLOR)
            #     cv2.imwrite(os.path.join('C:/Users/AI/PycharmProjects/class/new_classified/upper/', filename), org_img)

if __name__ == '__main__':
    #run_classifier()
    train()


