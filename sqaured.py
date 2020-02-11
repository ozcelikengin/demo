import numpy as np
import matplotlib.pyplot as plt
import os
import random
import cv2
import pickle


nDATADIR = "/Users/ozcelikengin/Desktop/all_signatures_squared"
os.remove(nDATADIR)
os.mkdir(nDATADIR) 
# os.makedirs("all_signatures_squared", exist_ok=True)

# locating dataset
# DATADIR = "/home/ozcelikengin/Desktop/demo/all_signatures"
DATADIR = "/Users/ozcelikengin/Documents/GitHub/demo/all_signatures"
CATEGORIES = ["You", "Hello", "Walk","Drink","Friend","Knife","Well","Car","Engineer","Mountain"]

# Image size for dataset
IMG_SIZE = 128 # default 128



training_data = []
test_data = []



def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        os.chdir(path)
        new_path = os.path.join(nDATADIR,category)
        os.mkdir(new_path)
        # os.makedirs(category,exist_ok=True)
        print(new_path)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                print(img)
                new_array = cv2.resize(img_array,(IMG_SIZE, IMG_SIZE))
                os.chdir(new_path)
                cv2.imwrite(img, new_array)
                # training_data.append([new_array, class_num])
            except Exception as e:
                print(e)
                pass
     


create_training_data()