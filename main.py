import configparser

import matplotlib.pyplot as plt
import numpy as np
import os

import sipconfig
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

import sys
from matplotlib import pyplot
import time
from tensorflow.python.keras.preprocessing.image import load_img
from tensorflow.python.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.models import load_model
import cv2
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tqdm import tqdm

import data_prepare
from model import *
from data_prepare import *


#import autokeras as ak

# dataset/
#       upper/
#       lower/
#       one_human/
#       multi_human/
#       cycle/
#       non_human/

# Press the green button in the gutter to run the script.

def summarize_diagnostics(history, model_name):
    fig, ax = plt.subplots(2, 2)

    #pyplot.subplot(211)
    pyplot = ax[0, 0]
    pyplot.set_title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.legend()
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    pyplot.legend()
    # plot accuracy
    #pyplot.subplot(221)
    pyplot = ax[1, 0]
    pyplot.set_title('Classification f1-score')
    pyplot.plot(history.history['f1_m'], color='blue', label='train')
    pyplot.legend()
    pyplot.plot(history.history['val_f1_m'], color='orange', label='test')
    pyplot.legend()

    #pyplot.subplot(212)
    pyplot = ax[0, 1]
    pyplot.set_title('Classification recall')
    pyplot.plot(history.history['recall_m'], color='blue', label='train')
    pyplot.legend()
    pyplot.plot(history.history['val_recall_m'], color='orange', label='test')
    pyplot.legend()

    #.subplot(222)
    pyplot = ax[1, 1]
    pyplot.set_title('Classification precision')
    pyplot.plot(history.history['precision_m'], color='blue', label='train')
    pyplot.legend()
    pyplot.plot(history.history['val_precision_m'], color='orange', label='test')
    pyplot.legend()
    fig.tight_layout()
    # save plot to file
    fig.savefig(model_name + "_" + time.strftime("%Y%m%d_%H%M%S") + '_plot' + '.png')

def train(model_name=None):
    model = globals()[model_name]()
    if model_name == "model_vgg16":
        Config.IMAGE_SIZE = 224
    elif model_name == "model_my1":
        Config.IMAGE_SIZE = 32
    elif model_name == "model_my2":
        Config.IMAGE_SIZE = 32
    elif model_name == "model_my3":
        Config.IMAGE_SIZE = 64
    elif model_name == "model_my4":
        Config.IMAGE_SIZE = 64
    print(Config.IMAGE_SIZE)

    print(model.summary())
    training_set, val_set, test_set = data_prepare.load_datasets()

    early_stopping = EarlyStopping()
    print(len(training_set))
    print(len(val_set))
    mc = ModelCheckpoint(model_name + '_bestmodel_' + time.strftime("%m%d_%H%M") + '.h5', monitor='val_f1_m', mode='max', save_best_only=True)
    history = model.fit(training_set,
                        steps_per_epoch=len(training_set),
                        epochs=Config.EPOCHS,
                        validation_data=val_set,
                        validation_steps=len(val_set),
                        verbose=1,
                        callbacks=[mc])
    print(history.history.keys())
    model.save('model_' + time.strftime("%Y%m%d_%H%M%S") + '.h5')
    print("-- Evaluate --")
    #_, acc = model.evaluate_generator(test_set, steps=len(test_set), verbose=0)
    #_, acc = model.evaluate(val_set, steps=len(val_set), verbose=2)
    k = model.evaluate(test_set, steps=len(test_set), verbose=0)
    print(k)
    loss, accuracy, f1_score, precision, recall = model.evaluate(test_set, steps=len(test_set), verbose=0)
    print('Accuracy: {} %'.format(round(accuracy * 100.0, 3)))
    print('f1_score: {} %'.format(f1_score))
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    print(test_set.class_indices)

    # test_set.reset()
    # print("-- Predict --")
    # output = model.predict_generator(val_set)
    # np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    # print(val_set.class_indices)
    # print(output)

    summarize_diagnostics(history, model_name = model_name)


def load_image (filename):
    img = load_img(filename, target_size=(Config.IMAGE_SIZE, Config.IMAGE_SIZE))
    #img = load_img(filename)
    img = img_to_array(img)
    img = img.reshape(1, Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3)
    #img = img.reshape((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3))
    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]
    return img


# batch norm 쓰고 가장 좋은거 bestmodel_20210715_165550
def run_classifier():

    # img = load_image('C:/Users/AI/PycharmProjects/class/datasets/test/upper/20210514_181704_person_2.jpg')
    # result = model.predict(img)
    # print(result[0])
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # cuda open success 안뜨게하기
    if os.path.isdir(Config.TEST_CYCLE):
        for file in os.scandir(Config.TEST_CYCLE):
            os.remove(file.path)
    elif not os.path.isdir(Config.TEST_CYCLE):
        os.mkdir(Config.TEST_CYCLE)

    if os.path.isdir(Config.TEST_MULTI):
        for file in os.scandir(Config.TEST_MULTI):
            os.remove(file.path)
    elif not os.path.isdir(Config.TEST_MULTI):
        os.mkdir(Config.TEST_MULTI)

    if os.path.isdir(Config.TEST_UPPER):
        for file in os.scandir(Config.TEST_UPPER):
            os.remove(file.path)
    elif not os.path.isdir(Config.TEST_UPPER):
        os.mkdir(Config.TEST_UPPER)

    if os.path.isdir(Config.TEST_LOWER):
        for file in os.scandir(Config.TEST_LOWER):
            os.remove(file.path)
    elif not os.path.isdir(Config.TEST_LOWER):
        os.mkdir(Config.TEST_LOWER)

    if os.path.isdir(Config.TEST_NON):
        for file in os.scandir(Config.TEST_NON):
            os.remove(file.path)
    elif not os.path.isdir(Config.TEST_NON):
        os.mkdir(Config.TEST_NON)

    if os.path.isdir(Config.TEST_ONE):
        for file in os.scandir(Config.TEST_ONE):
            os.remove(file.path)
    elif not os.path.isdir(Config.TEST_ONE):
        os.mkdir(Config.TEST_ONE)
    model = load_model(Config.MODEL_NAME, custom_objects={'f1_m': f1_m,
                                                          'precision_m': precision_m,
                                                          'recall_m': recall_m
                                                          })
    #model = load_model(Config.MODEL_NAME, compile=True, custom_objects={'get_f1': get_f1})
    predict_dataset = Config.VALIDATION_DIR
    iter=0
    for filename in tqdm(os.listdir(predict_dataset)):
        iter += 1
        #print("progress: {}, filename: {} ".format(progress, filename))
        img = load_image(os.path.join(predict_dataset, filename))
        #result = model.predict_generator(img)
        result = model.predict(img)
        #print(iter, result[0])

        # 헷갈려하는 사진들이 뭔지 찾기
        # if np.max(result[0]) < 0.8:
        #     org_img = cv2.imread(os.path.join(predict_dataset, filename), cv2.IMREAD_COLOR)
        #     cv2.imwrite(os.path.join('dontknow/', filename), org_img)


        if np.argmax(result[0]) == 0:
            org_img = cv2.imread(os.path.join(predict_dataset, filename), cv2.IMREAD_COLOR)
            cv2.imwrite(os.path.join(Config.TEST_CYCLE, filename), org_img)
        elif np.argmax(result[0]) == 1:
            org_img = cv2.imread(os.path.join(predict_dataset, filename), cv2.IMREAD_COLOR)
            cv2.imwrite(os.path.join(Config.TEST_LOWER, filename), org_img)
        elif np.argmax(result[0]) == 2:
            org_img = cv2.imread(os.path.join(predict_dataset, filename), cv2.IMREAD_COLOR)
            cv2.imwrite(os.path.join(Config.TEST_MULTI, filename), org_img)
        elif np.argmax(result[0]) == 3:
            org_img = cv2.imread(os.path.join(predict_dataset, filename), cv2.IMREAD_COLOR)
            cv2.imwrite(os.path.join(Config.TEST_NON, filename), org_img)
        elif np.argmax(result[0]) == 4:
            org_img = cv2.imread(os.path.join(predict_dataset, filename), cv2.IMREAD_COLOR)
            cv2.imwrite(os.path.join(Config.TEST_ONE, filename), org_img)
        elif np.argmax(result[0]) == 5:
            org_img = cv2.imread(os.path.join(predict_dataset, filename), cv2.IMREAD_COLOR)
            cv2.imwrite(os.path.join(Config.TEST_UPPER, filename), org_img)

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
        #
        #     cv2.imwrite(os.path.join('C:/Users/AI/PycharmProjects/class/new_classified/upper/', filename), org_img)

if __name__ == '__main__':
    #run_classifier()
    train(model_name="model_my1")
    #train(model_name="model_my2")
    # train(model_name="model_my3")
    # train(model_name="model_my4")
    #train(model_name="model_vgg16")

