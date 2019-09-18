import cv2
import numpy as np
import random as rd


class TransformUtils:

    def __init__(self, h, w):
        self.HEIGHT = h
        self.WIDTH = w

    def darken(self, img):
        coeff = rd.randint(4, 8)/10
        return (img*coeff).astype(np.uint8)

    def brighten(self, img):
        coeff = rd.randint(12, 16)/10
        v2 = 2 - coeff
        return (img * v2 + (1 - v2) * 255).astype(np.uint8)

    def rotation(self, img):
        rotation_angle = np.random.normal(0, 0.7)
        T = cv2.getRotationMatrix2D((self.HEIGHT/2, self.WIDTH/2), rotation_angle, 1)
        return cv2.warpAffine(img, T, (self.WIDTH, self.HEIGHT), borderMode= cv2.BORDER_CONSTANT, borderValue= 255)

    def updown(self, img):
        return np.flipud(img)

    def translation(self, img):
        tx, ty = np.random.normal(0, 2.25, 2)
        T = np.float32([[1, 0, tx], [0, 1, ty]])
        return cv2.warpAffine(img, T, (self.WIDTH, self.HEIGHT), borderMode= cv2.BORDER_CONSTANT, borderValue= 255)

    def occlusion(self, img):
        total_area = self.WIDTH*self.HEIGHT
        r = rd.randint(0, self.HEIGHT-10)
        h = rd.randint(20, int(self.HEIGHT/3))
        percent_occluded_area = rd.randint(15, 35)/100
        w = min(self.WIDTH, int(total_area*percent_occluded_area/h))
        c = rd.randint(0, self.WIDTH - w)
        img[r:r + h, c:c + w] = 255
        return img

T = TransformUtils(100, 196)


# for k in range(50):
#     blank_image = np.ones((100, 196, 1), np.uint8)*100
#     im = T.brighten(blank_image)
#     cv2.imshow("im", im)
#     cv2.waitKey(500)

