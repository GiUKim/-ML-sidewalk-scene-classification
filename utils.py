import configparser

import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
# import os
# from glob import glob
# import numpy as np
# from tqdm import tqdm
# import cv2
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from data_prepare import *
from model import *
from config import *
import time
import os

def summarize_diagnostics(history):
    fig, ax = plt.subplots(2, 2)

    #pyplot.subplot(211)
    pyplot = ax[0, 0]
    pyplot.set_title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.legend()
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    pyplot.legend()

    pyplot = ax[1, 0]
    pyplot.set_title('Classification f1-score')
    pyplot.plot(history.history['f1'], color='blue', label='train')
    pyplot.legend()
    pyplot.plot(history.history['val_f1'], color='orange', label='test')
    pyplot.legend()

    #pyplot.subplot(212)
    pyplot = ax[0, 1]
    pyplot.set_title('Classification recall')
    pyplot.plot(history.history['recall'], color='blue', label='train')
    pyplot.legend()
    pyplot.plot(history.history['val_recall'], color='orange', label='test')
    pyplot.legend()

    #.subplot(222)
    pyplot = ax[1, 1]
    pyplot.set_title('Classification precision')
    pyplot.plot(history.history['precision'], color='blue', label='train')
    pyplot.legend()
    pyplot.plot(history.history['val_precision'], color='orange', label='test')
    pyplot.legend()
    fig.tight_layout()
    # save plot to file
    fig.savefig(time.strftime("%Y%m%d_%H%M%S") + '_plot' + '.png')

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def train(exist_data = True, model=None):
    if not exist_data:
        x_train, y_train, x_test, y_test = data_prepare()
    else:
        x_train = np.load('x_train.npy')
        y_train = np.load('y_train.npy')
        x_test = np.load('x_test.npy')
        y_test = np.load('y_test.npy')

    #model = Model3()
    #model = VGG16()
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='binary_crossentropy',
                  metrics=['accuracy', f1, precision, recall])

    mc = ModelCheckpoint('bestmodel_'+time.strftime("%m%d_%H%M")+'.h5', monitor='val_f1', mode='max', save_best_only=True)
    hist = model.fit(x_train, y_train,
                     batch_size=Config.BATCH_SIZE,
                     shuffle=True,
                     epochs=Config.EPOCHS,
                     validation_data=(x_test, y_test),
                     callbacks=[mc])
    summarize_diagnostics(hist)
    model.evaluate(x_test, y_test, batch_size=1)
    model.save('model_' + time.strftime("%Y%m%d_%H%M%S") + '.h5')

def clean_new_classified_folder():
    if os.path.isdir(Config.test_save_cycle):
        for file in os.scandir(Config.test_save_cycle):
            os.remove(file.path)
    elif not os.path.isdir(Config.test_save_cycle):
        os.mkdir(Config.test_save_cycle)

    if os.path.isdir(Config.test_save_multi):
        for file in os.scandir(Config.test_save_multi):
            os.remove(file.path)
    elif not os.path.isdir(Config.test_save_multi):
        os.mkdir(Config.test_save_multi)

    if os.path.isdir(Config.test_save_upper):
        for file in os.scandir(Config.test_save_upper):
            os.remove(file.path)
    elif not os.path.isdir(Config.test_save_upper):
        os.mkdir(Config.test_save_upper)

    if os.path.isdir(Config.test_save_lower):
        for file in os.scandir(Config.test_save_lower):
            os.remove(file.path)
    elif not os.path.isdir(Config.test_save_lower):
        os.mkdir(Config.test_save_lower)

    if os.path.isdir(Config.test_save_non):
        for file in os.scandir(Config.test_save_non):
            os.remove(file.path)
    elif not os.path.isdir(Config.test_save_non):
        os.mkdir(Config.test_save_non)

    if os.path.isdir(Config.test_save_one):
        for file in os.scandir(Config.test_save_one):
            os.remove(file.path)
    elif not os.path.isdir(Config.test_save_one):
        os.mkdir(Config.test_save_one)

def predict_and_save():
    clean_new_classified_folder()
    model = load_model(Config.MODEL_NAME, custom_objects={'f1': f1,
                                                          'precision': precision,
                                                          'recall': recall
                                                          })

    for filename in tqdm(os.listdir(Config.test_dir)):
        img = Image.open(os.path.join(Config.test_dir, filename))

        test_img = img.resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE))
        test_img = np.array(test_img)
        test_img = test_img / 255.
        pred = model.predict(test_img[tf.newaxis, ...])
        if np.all(pred < Config.threshold):
            plt.imsave(os.path.join(Config.test_save_non, filename), np.array(img))
        elif np.argmax(pred) == 0:
            plt.imsave(os.path.join(Config.test_save_cycle, filename), np.array(img))
        elif np.argmax(pred) == 1:
            plt.imsave(os.path.join(Config.test_save_lower, filename), np.array(img))
        elif np.argmax(pred) == 2:
            plt.imsave(os.path.join(Config.test_save_multi, filename), np.array(img))
        elif np.argmax(pred) == 3:
            plt.imsave(os.path.join(Config.test_save_one, filename), np.array(img))
        elif np.argmax(pred) == 4:
            plt.imsave(os.path.join(Config.test_save_upper, filename), np.array(img))

def evaluate():
    x_test = np.load('x_test.npy')
    y_test = np.load('y_test.npy')
    model = load_model(Config.MODEL_NAME, custom_objects={'f1': f1,
                                                          'precision': precision,
                                                          'recall': recall
                                                          })
    print(model.summary())
    cnt_list = [0] * 6
    ans_list = [0] * 6
    for n in y_test:
        if np.all(n == ((0., 0., 0., 0., 0.))):
            cnt_list[5] += 1
        elif np.all(n == ((1., 0., 0., 0., 0.))):
            cnt_list[0] += 1
        elif np.all(n == ((0., 1., 0., 0., 0.))):
            cnt_list[1] += 1
        elif np.all(n == ((0., 0., 1., 0., 0.))):
            cnt_list[2] += 1
        elif np.all(n == ((0., 0., 0., 1., 0.))):
            cnt_list[3] += 1
        elif np.all(n == ((0., 0., 0., 0., 1.))):
            cnt_list[4] += 1

    for idx in tqdm(range(len(x_test))):
        pred = model.predict(x_test[idx][tf.newaxis, ...])
        if np.all(pred < Config.threshold) and np.all(y_test[idx] == ((0., 0., 0., 0., 0.))):
            ans_list[5] += 1
        elif np.any(pred >= Config.threshold):
            if np.argmax(pred) == 0 and np.all(y_test[idx] == ((1., 0., 0., 0., 0.))):
                ans_list[0] += 1
            elif np.argmax(pred) == 1 and np.all(y_test[idx] == ((0., 1., 0., 0., 0.))):
                ans_list[1] += 1
            elif np.argmax(pred) == 2 and np.all(y_test[idx] == ((0., 0., 1., 0., 0.))):
                ans_list[2] += 1
            elif np.argmax(pred) == 3 and np.all(y_test[idx] == ((0., 0., 0., 1., 0.))):
                ans_list[3] += 1
            elif np.argmax(pred) == 4 and np.all(y_test[idx] == ((0., 0., 0., 0., 1.))):
                ans_list[4] += 1
        #print(pred)
    print("Cycle Accuracy: {} %".format(round(100. * ans_list[0] / cnt_list[0], 2)))
    print("Lower Accuracy: {} %".format(round(100. * ans_list[1] / cnt_list[1], 2)))
    print("Multi Accuracy: {} %".format(round(100. * ans_list[2] / cnt_list[2], 2)))
    print("One Accuracy: {} %".format(round(100. * ans_list[3] / cnt_list[3], 2)))
    print("Upper Accuracy: {} %".format(round(100. * ans_list[4] / cnt_list[4], 2)))
    print("Non Accuracy: {} %".format(round(100. * ans_list[5] / cnt_list[5], 2)))
    print("\nTOTAL Accuracy: {} %".format(round(100. * np.sum(ans_list) / np.sum(cnt_list), 2)))

