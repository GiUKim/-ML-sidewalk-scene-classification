import configparser

import tensorflow as tf
from PIL import Image

from glob import glob
import numpy as np
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical

from config import *
def data_prepare():
    num_classes = Config.NUM_CLASSES
    cycle_train_dir = Config.cycle_train_dir
    lower_train_dir = Config.lower_train_dir
    multi_train_dir = Config.multi_train_dir
    one_train_dir = Config.one_train_dir
    upper_train_dir = Config.upper_train_dir
    non_train_dir = Config.non_train_dir

    cycle_test_dir = Config.cycle_test_dir
    lower_test_dir = Config.lower_test_dir
    multi_test_dir = Config.multi_test_dir
    one_test_dir = Config.one_test_dir
    upper_test_dir =Config.upper_test_dir
    non_test_dir = Config.non_test_dir

    def load_img_to_list(img_path):
        list = []
        for img in tqdm(img_path):
            #img = Image.open(img).convert('L')
            img = Image.open(img)
            img = img.resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE), Image.ANTIALIAS)
            img_np = np.array(img)
            img.close()
            img_np = img_np / 255.
            list.append(img_np)

        return list

    cycle_train_list = np.array(load_img_to_list(cycle_train_dir))
    lower_train_list = np.array(load_img_to_list(lower_train_dir))
    multi_train_list = np.array(load_img_to_list(multi_train_dir))
    one_train_list = np.array(load_img_to_list(one_train_dir))
    upper_train_list = np.array(load_img_to_list(upper_train_dir))
    non_train_list = np.array(load_img_to_list(non_train_dir))

    cycle_test_list = np.array(load_img_to_list(cycle_test_dir))
    lower_test_list = np.array(load_img_to_list(lower_test_dir))
    multi_test_list = np.array(load_img_to_list(multi_test_dir))
    one_test_list = np.array(load_img_to_list(one_test_dir))
    upper_test_list = np.array(load_img_to_list(upper_test_dir))
    non_test_list = np.array(load_img_to_list(non_test_dir))

    x_train = np.append(cycle_train_list, lower_train_list, 0)
    x_train = np.append(x_train, multi_train_list, 0)
    x_train = np.append(x_train, one_train_list, 0)
    x_train = np.append(x_train, upper_train_list, 0)
    x_train = np.append(x_train, non_train_list, 0)
    #x_train = x_train[..., tf.newaxis]

    x_test = np.append(cycle_test_list, lower_test_list, 0)
    x_test = np.append(x_test, multi_test_list, 0)
    x_test = np.append(x_test, one_test_list, 0)
    x_test = np.append(x_test, upper_test_list, 0)
    x_test = np.append(x_test, non_test_list, 0)
    #x_test = x_test[..., tf.newaxis]

    # encoding
    y_train = []
    for i in cycle_train_list:
        y_train.append(to_categorical(0, num_classes))
    for i in lower_train_list:
        y_train.append(to_categorical(1, num_classes))
    for i in multi_train_list:
        y_train.append(to_categorical(2, num_classes))
    for i in one_train_list:
        y_train.append(to_categorical(3, num_classes))
    for i in upper_train_list:
        y_train.append(to_categorical(4, num_classes))
    for i in non_train_list:
        y_train.append(np.array((0, 0, 0, 0, 0)))
    y_train = np.array(y_train)

    y_test = []
    for i in cycle_test_list:
        y_test.append(to_categorical(0, num_classes))
    for i in lower_test_list:
        y_test.append(to_categorical(1, num_classes))
    for i in multi_test_list:
        y_test.append(to_categorical(2, num_classes))
    for i in one_test_list:
        y_test.append(to_categorical(3, num_classes))
    for i in upper_test_list:
        y_test.append(to_categorical(4, num_classes))
    for i in non_test_list:
        y_test.append(np.array((0, 0, 0, 0, 0)))
    y_test = np.array(y_test)
    np.save(Config.base_dir + 'x_train', x_train)
    np.save(Config.base_dir + 'y_train', y_train)
    np.save(Config.base_dir + 'x_test', x_test)
    np.save(Config.base_dir + 'y_test', y_test)

    return x_train, y_train, x_test, y_test

