import cv2
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
picture = cv2.imread('lena.jpg',cv2.IMREAD_COLOR)
# ---------------------------------------------------------------- #
def rotate_bound1(image, angle):
    (h, w) = image.shape[:2]  # 返回(高,寬,色彩通道数),此取前兩個值
    # 抓取旋转矩阵(应用角度的负值顺时针旋转)
    # cv2.getRotationMatrix2D(旋轉中心點,旋轉角度(逆為正),同性比例因子)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)
    print(M)
    # 計算新圖像的邊長([2*3]*[3*1]=[2*1] --> 但這邊用不到channel??:為什麼是3*3? )-->([2*2]*[2*1]=[2*1])
    newW = int((h * np.abs(M[0, 1])) + (w * np.abs(M[0, 0])))
    newH = int((h * np.abs(M[1, 0])) + (w * np.abs(M[1, 1])))
    print(M[0, 0])
    print(M[0, 1])
    print(M[1, 0])
    print(M[1, 1])
    # 调整旋转矩阵以考虑平移
    # c += a 相當於 c = c + a
    M[0, 2] += (newW - w) / 2
    M[1, 2] += (newH - h) / 2
    print(M)
    # 执行实际的旋转并返回图像
    return cv2.warpAffine(image, M, (newW, newH)) # 圖片外的區域，默认是黑色
# ---------------------------------------------------------------- #
picture = rotate_bound1(picture, -45)
picture_scale =  cv2.resize(picture, None, fx=0.5, fy=0.5,interpolation=cv2.INTER_AREA)
def translate(image, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifte = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifte
shifte1 = translate(picture_scale, 100,-50)
cv2.imshow('Translation', shifte1)
cv2.waitKey(0)
cv2.destroyAllWindows()