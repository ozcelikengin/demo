import numpy as np
import matplotlib.pyplot as plt
import os
import random
import cv2
import pickle

# locating dataset
DATADIR = "/home/ozcelikengin/Desktop/demo/all_signatures"
CATEGORIES = ["You", "Hello", "Walk","Drink","Friend","Knife","Well","Car","Engineer","Mountain"]

# Image size for dataset
IMG_SIZE = 128


training_data = []
test_data = []


def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array,(IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                print(e)
                pass

    random.shuffle(training_data)
    


    X = np.array(X).reshape(-1,IMG_SIZE, IMG_SIZE)


    pickle_out = open("X.pickle", "wb")
    pickle.dump(X,pickle_out)
    pickle_out.close()

    pickle_out = open("y.pickle", "wb")
    pickle.dump(y,pickle_out)
    pickle_out.close()

    pickle_in = open("x.pickle", "rb")
    X = pickle.load(pickle_in)


    return X,y 

    # Update