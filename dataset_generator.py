import barcode
import random as rd
import numpy as np
import os
import glob
import cv2
from barcode.writer import ImageWriter
from tensorflow.python.keras.utils import to_categorical

from parameters import *
from transform_utils import TransformUtils

#First, delete everything in ./dataset
cwd = os.getcwd()
path = os.path.join(cwd, "dataset")

os.chdir(path)
for image_path in glob.glob("*.png"):
    file_path = os.path.join(path, image_path)
    os.unlink(file_path)
if os.path.exists(os.path.join(path, 'labels.txt')) :
    os.unlink(os.path.join(path, 'labels.txt'))
os.chdir("../")
coder = barcode.get_barcode_class(barcode_type)


T = TransformUtils(w=WIDTH, h=HEIGHT)
labels = np.zeros((nb_img, 100), dtype=np.bool_)
for k in range(nb_img):
    nb = rd.randint(0, max_barcode_number)
    nb_digits = len(str(max_barcode_number))
    nb = str(nb).zfill(nb_digits)
    barcode  = coder(nb, ImageWriter())
    barcode.save('./dataset/temp', options=barcode_options)
    temp = cv2.imread("./dataset/temp.png", cv2.IMREAD_GRAYSCALE)
    temp = cv2.resize(temp, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    ##########################################################
    ################### DATA AUGMENTATION ####################
    ##########################################################
    if rd.random()<0.5 :
        temp = T.updown(temp)
    temp = T.occlusion(temp)
    temp = T.translation(temp)
    temp = T.rotation(temp)
    if rd.random()<0.66:
        if rd.random()<0.5:
            temp = T.darken(temp)
        else:
            temp = T.brighten(temp)
    ##########################################################
    cv2.imwrite(os.path.join(path, "cb-" + str(k) + ".png"), temp)

    nb_arr = np.array(list(nb))
    vect = []
    for d in nb_arr:
        cat = to_categorical(d, 10)
        vect.append(cat)
    # mat = np.stack(vect)
    # labels[k*10:k*10+10, :]  = mat
    vect = np.concatenate(vect, axis=0)
    labels[k, :] = vect
np.save("./dataset/labels", labels)
os.unlink("./dataset/temp.png")
