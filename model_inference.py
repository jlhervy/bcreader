import cv2
from tensorflow.keras.models import load_model
from parameters import *
import numpy as np
import models

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

def compare_diff(res, truth):
    truth = truth.astype(np.float32)
    nb_diff = 0
    for k in range(nb_digits):
        if not((truth[k*nb_classes:k*nb_classes+10] == res[0][k*nb_classes:k*nb_classes+10]).all()):
            nb_diff = nb_diff +1
    print(nb_diff)
y = np.load("./inference/labels.npy")

model = load_model("./logs/train-41/best_model.h5")
for i in range(50):
    cb = cv2.imread("./inference/cb-" + str(i) + ".png", cv2.IMREAD_GRAYSCALE)
    cb = cb.reshape([1, HEIGHT, WIDTH, 1])
    res = model.predict(cb)
    truth = y[i]
    compare_diff(res, truth)