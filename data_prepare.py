
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from config import *

def load_datasets():
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           validation_split=0.1
                                           )

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
    return training_set, val_set