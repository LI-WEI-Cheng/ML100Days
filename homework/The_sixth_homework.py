import cv2
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
picture = cv2.imread('lena.jpg',cv2.IMREAD_COLOR)
# ---------------------------------------------------------------- #
(h, w) = picture.shape[:2]
M = cv2.getRotationMatrix2D((w / 2, h / 2), 45, 1.0)
print(M)
newW = int((h * np.abs(M[0, 1])) + (w * np.abs(M[0, 0])))
newH = int((h * np.abs(M[1, 0])) + (w * np.abs(M[1, 1])))
print(M[0, 0])
print(M[0, 1])
print(M[1, 0])
print(M[1, 1])
M[0, 2] += (newW - w) / 2
M[1, 2] += (newH - h) / 2
print(M)
picture = cv2.warpAffine(picture, M, (newW, newH))
# ---------------------------------------------------------------- #
picture_scale =  cv2.resize(picture, None, fx=0.5, fy=0.5,interpolation=cv2.INTER_AREA)
M123 = np.float32([[1, 0, 100], [0, 1, -50]])
shifte = cv2.warpAffine(picture_scale, M123, (picture_scale.shape[1], picture_scale.shape[0]))
cv2.imshow('Translation', shifte)
cv2.waitKey(0)
cv2.destroyAllWindows()