import cv2
import numpy as np
img = cv2.imread('lena.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
SIFT_detector = cv2.xfeatures2d.SIFT_create()
keypoints6 = SIFT_detector.detect(img_gray, None)
img_show = cv2.drawKeypoints(img_gray, keypoints6, img)
# picture_123 = np.hstack((picture,img_show ))
cv2.imshow('SIFT', img_show )
cv2.waitKey(0)
cv2.destroyAllWindows()
# -------------------------------------------------------------------- #
picture = cv2.imread('lena.jpg')
picture_gray = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
SIFT = cv2.xfeatures2d.SIFT_create()
keypoints1 = SIFT.detect(picture_gray, None)
picture_show = cv2.drawKeypoints(picture_gray, keypoints1, picture)
picture_123 = np.hstack((picture_show,img_show ))

cv2.imshow('SIFT', picture_123 )

cv2.waitKey(0)
cv2.destroyAllWindows()