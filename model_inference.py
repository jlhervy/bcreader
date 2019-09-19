import cv2
from tensorflow.python.keras.models import load_model
from parameters import *
import numpy as np

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

y = np.load("./inference/labels.npy")

model = load_model("./logs/train-1/best_model.h5")
for i in range(50):
    cb = cv2.imread("./inference/cb-" + str(i) + ".png", cv2.IMREAD_GRAYSCALE)
    cb = cb.reshape([1, HEIGHT, WIDTH, 1])
    res = model.predict(cb)
    truth = [y[k][i] for k in range(10)]
    print(from_categorical_to_integers(res))
    print(from_categorical_to_integers(truth))