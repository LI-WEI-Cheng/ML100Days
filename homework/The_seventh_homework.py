import cv2
import numpy as np
from PIL import Image, ImageFilter
picture = cv2.imread('lena.jpg',cv2.IMREAD_COLOR)
picture_gray = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
picture_for_blur = picture.copy()
picture_blur = cv2.GaussianBlur(picture,(3,3),0)
picture_blur = cv2.GaussianBlur(picture_blur,(3,3),0)
picture_blur1= cv2.GaussianBlur(picture_blur,(3,3),0)
picture_blur2 = cv2.GaussianBlur(picture_blur1,(3,3),0)
picture_blur2 = cv2.GaussianBlur(picture_blur2,(3,3),0)
cv2.imshow('BLUR', picture_blur)
picture_blur = np.hstack((picture, picture_blur1,picture_blur2))
cv2.imshow('BLUR', picture_blur)
cv2.waitKey(0)
cv2.destroyAllWindows()
picture_x = cv2.Sobel(picture_gray,cv2.CV_16S,dx = 1,dy=0,ksize=3)
picture_y = cv2.Sobel(picture_gray,cv2.CV_16S,dx = 0,dy=1,ksize=3)
picture_x = cv2.convertScaleAbs(picture_x)
picture_y = cv2.convertScaleAbs(picture_y)
picture_sobel1 = cv2.addWeighted(picture_x,0.5,picture_y,0.5,0)
# ------------------------------------------------------------- #
picture_sobel2 = cv2.addWeighted(picture_x,1,picture_y,0,0)
# ------------------------------------------------------------- #
picture_sobel3 = cv2.addWeighted(picture_x,0,picture_y,2,0)
# ------------------------------------------------------------- #
picture_sobel4 = cv2.addWeighted(picture_x,0.5,picture_y,0.5,100)
# ------------------------------------------------------------- #
picture_show_sobel = np.hstack((picture_sobel1, picture_sobel2,picture_sobel3,picture_sobel4))
cv2.imshow('SOBEL', picture_show_sobel )
cv2.waitKey(0)
cv2.destroyAllWindows()