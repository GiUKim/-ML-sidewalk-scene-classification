import cv2
from tqdm import tqdm
import os
import numpy as np
from config import Config
import imutils
inst = {'flip': 0,
        'GaussianBlur': 0,
        'brightness': 1,
        'rotate': 0
       }

# metric
remove_augmentation = False
brightness = 10
rotate_angle = 10
gaussian_kernel = 5
class_name = 'multi_human' # upper
iter = 0
directory = Config.base_dir + 'datasets/train/'+class_name
file_list = [filenames for (filenames) in os.listdir(directory)]
file_list_jpg = [file for file in file_list if file.endswith(".jpg")]
if not remove_augmentation:
    for filename in tqdm(os.listdir(directory)):
        iter += 1
        if iter >6500:
            break
        img_path = os.path.join(directory, filename)
        size = os.path.getsize(img_path)
        org = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if size <= 0:
            os.remove(img_path)
            continue
        else:
            if inst['flip']:
                # flip
                gen = cv2.flip(org, 1)
                cv2.imwrite(Config.base_dir + "datasets/train/"+class_name+"/AUG_FLIP_"+filename, gen)
            if inst['GaussianBlur']:
                # Gaussian blur
                gen = cv2.GaussianBlur(org, (gaussian_kernel, gaussian_kernel), 0)
                cv2.imwrite(Config.base_dir + "datasets/train/"+class_name+"/AUG_GAUSS_"+filename, gen)
            if inst['brightness']:
                # brightness up
                arr = np.full(org.shape, (brightness, brightness, brightness), dtype=np.uint8)
                gen = cv2.add(org, arr)
                cv2.imwrite(Config.base_dir + "datasets/train/"+class_name+"/AUG_UP_"+filename, gen)
            if inst['rotate']:
                # rotation
                gen = imutils.rotate(org, rotate_angle)
                cv2.imwrite(Config.base_dir + "datasets/train/"+class_name+"/AUG_ROT_"+filename, gen)
else:
    file_list_aug = [file for file in file_list if file.startswith("AUG")]
    for filename in tqdm(file_list_aug):
        img_path = os.path.join(directory, filename)
        os.remove(img_path)