import cv2
from tqdm import tqdm
import os
import numpy as np
inst = {'flip': 1,
        'GaussianBlur': 1,
        'brightness': 1
       }
remove_augmentation = False

class_name = 'upper'

directory = 'C:/Users/AI/PycharmProjects/class/datasets/train/'+class_name
file_list = [filenames for (filenames) in os.listdir(directory)]
file_list_jpg = [file for file in file_list if file.endswith(".jpg")]
if not remove_augmentation:
    for filename in tqdm(file_list_jpg):
        img_path = os.path.join(directory, filename)
        size = os.path.getsize(img_path)
        if size <= 0:
            os.remove(img_path)
            continue
        else:
            org = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if inst['flip']:
                # flip
                #org_cvt = cv2.cvtColor(org, cv2.COLOR_BGR2RGB)
                gen = cv2.flip(org, 1)
                cv2.imwrite("C:/Users/AI/PycharmProjects/class/datasets/train/"+class_name+"/AUG_FLIP_"+filename, gen)
            if inst['GaussianBlur']:
                # Gaussian blur
                gen = cv2.GaussianBlur(org, [5, 5], 0)
                cv2.imwrite("C:/Users/AI/PycharmProjects/class/datasets/train/"+class_name+"/AUG_GAUSS_"+filename, gen)
            if inst['brightness']:
                # brightness up, down
                arr = np.full(org.shape, (40, 40, 40), dtype=np.uint8)
                gen = cv2.add(org, arr)
                gen2 = cv2.subtract(org, arr)
                cv2.imwrite("C:/Users/AI/PycharmProjects/class/datasets/train/"+class_name+"/AUG_UP_"+filename, gen)
                cv2.imwrite("C:/Users/AI/PycharmProjects/class/datasets/train/"+class_name+"/AUG_DOWN_"+filename, gen2)

else:
    file_list_aug = [file for file in file_list if file.startswith("AUG")]
    for filename in tqdm(file_list_aug):
        img_path = os.path.join(directory, filename)
        os.remove(img_path)