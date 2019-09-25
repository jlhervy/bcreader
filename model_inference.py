import cv2
from tensorflow.keras.models import load_model
from parameters import *
import numpy as np
import models
import tensorflow as tf

model_path = "./logs/train-8/best_model.h5"

def from_categorical_to_integers(res):
    nb = ""
    for k in range(res.__len__()):
        if (res[0].shape.__len__() == 2) :
            cat = res[k][0]
        else :
            cat = res[k]
        dg = np.where(cat==1)[0][0]
        nb = nb + str(dg)
    return nb

def compare_diff(res):
    nb_diff = 0
    for k in range(nb_digits):
        if not((y[k][i] == res[0][k*nb_classes:k*nb_classes+10]).all()):
            nb_diff = nb_diff +1
    return nb_diff
y = np.load("./inference/labels.npy")
y = y.astype(np.float32)
with tf.device('/gpu:1'):
    model = load_model(model_path)
    total_diff = 0
    for i in range(1000):
        cb = cv2.imread("./inference/cb-" + str(i) + ".png", cv2.IMREAD_GRAYSCALE)
        cb = cb.reshape([1, HEIGHT, WIDTH, 1])
        res = model.predict(cb)
        nb_diff = compare_diff(res)
        total_diff = total_diff + nb_diff
    print("total nb of wrong digits : " + str(total_diff))