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
path = os.path.join(cwd, generation_folder)

os.chdir(path)
for image_path in glob.glob("*.png"):
    file_path = os.path.join(path, image_path)
    os.unlink(file_path)
if os.path.exists(os.path.join(path, 'labels.txt')) :
    os.unlink(os.path.join(path, 'labels.txt'))
os.chdir("../")
coder = barcode.get_barcode_class(barcode_type)


T = TransformUtils(w=WIDTH, h=HEIGHT)
labels = np.array([np.zeros((nb_img, 10), dtype=np.bool_)] *10)
for k in range(nb_img):
    nb = rd.randint(0, max_barcode_number)
    nb_digits = len(str(max_barcode_number))
    nb = str(nb).zfill(nb_digits)
    barcode  = coder(nb, ImageWriter(), add_checksum=False)
    barcode.save("./" +generation_folder+'/temp', options=barcode_options)
    temp = cv2.imread("./" +generation_folder+'/temp.png', cv2.IMREAD_GRAYSCALE)
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
    for d in range(nb_arr.shape[0]):
        cat = to_categorical(nb_arr[d], 10)
        labels[d][k] = cat
np.save("./" + generation_folder + "/labels", labels)
os.unlink("./" + generation_folder + "/temp.png")
