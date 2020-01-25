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
IMG_SIZE = 28 # default 128



training_data = []
test_data = []


def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path)[:-1]:
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array,(IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                print(e)
                pass
        for img_test in os.listdir(path)[-1:]:
            try:
                img_test_array = cv2.imread(os.path.join(path,img_test), cv2.IMREAD_GRAYSCALE)
                new_test_array = cv2.resize(img_test_array,(IMG_SIZE, IMG_SIZE))
                test_data.append([new_test_array, class_num])
            except Exception as e_test:
                print(e_test)
                pass

   # random.shuffle(training_data)
    X = [] # Feature
    X_t = [] # Test feature
    y = [] # Label
    y_t = [] # Test label

    for features,label in training_data:
        X.append(features)
        y.append(label)
    for features_test,label_test in test_data:
        X_t.append(features_test)
        y_t.append(label_test)

    X = np.array(X).reshape(-1,IMG_SIZE, IMG_SIZE)
    X_t = np.array(X_t).reshape(-1,IMG_SIZE, IMG_SIZE)
    


    pickle_out = open("X.pickle", "wb")
    pickle.dump(X,pickle_out)
    pickle_out.close()

    pickle_out = open("y.pickle", "wb")
    pickle.dump(y,pickle_out)
    pickle_out.close()
    
    pickle_out = open("X_t.pickle", "wb")
    pickle.dump(X_t,pickle_out)
    pickle_out.close()

    pickle_out = open("y_t.pickle", "wb")
    pickle.dump(y_t,pickle_out)
    pickle_out.close()

    pickle_in = open("X.pickle", "rb")
    X = pickle.load(pickle_in)
    pickle_in.close()
    
    pickle_in = open("X_t.pickle", "rb")
    X_t = pickle.load(pickle_in)
    pickle_in.close()

    return (X, y), (X_t, y_t)
