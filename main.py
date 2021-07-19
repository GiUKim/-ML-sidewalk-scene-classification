import numpy as np
import os
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
from model import *


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
    pyplot.savefig(model_name + "_" + time.strftime("%Y%m%d_%H%M%S") + '_plot' + '.png')
    pyplot.close()

def train(model_name=None):
    #model = model_structure()
    if model_name == "org":
        model = model_org()
    elif model_name == "res1":
        model = model_myres1()
    elif model_name == "res2":
        model = model_myres2()
    else:
        model = model_vgg16()

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

    summarize_diagnostics(history, model_name = model_name)
    return round(acc * 100.0, 3)

def load_image(filename):
    img = load_img(filename, target_size=(Config.IMAGE_SIZE, Config.IMAGE_SIZE))
    #img = load_img(filename)
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

    predict_dataset = Config.TEST_DIR
    for filename in tqdm(os.listdir(predict_dataset)):
        if filename.endswith(".jpg"):
            progress += 1
            #print("progress: {}, filename: {} ".format(progress, filename))
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
    train(model_name="org")
    train(model_name="res1")
    train(model_name="res2")
    train(model_name="vgg16")


