import cv2
# 透過 imread 載入圖片，載入包含 Blue, Green, Red 三個 channel 的彩色圖片
picture = cv2.imread('unnamed.jpg',cv2.IMREAD_COLOR)
img2 = cv2.cvtColor(picture, cv2.COLOR_BGR2HLS)
img3 = cv2.cvtColor(picture, cv2.COLOR_BGR2LAB)
cv2.imshow('rgb2', img2)
cv2.imshow('rgb3', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()