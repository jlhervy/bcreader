import numpy as np
import cv2
import os
from tensorflow.python.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from parameters import *


cwd = os.getcwd()
path = os.path.join(cwd, "dataset")

all_images = []
for i in range(nb_img):
    image_path = os.path.join(path, "cb-" + str(i) + ".png")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = img.reshape([HEIGHT, WIDTH, 1])
    all_images.append(img)


X = np.array(all_images)
X = X/255.0

y = np.load("./dataset/labels.npy")
y = [y[k] for k in range(10)]

