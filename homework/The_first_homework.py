import cv2

img_path = 'unnamed.jpg'

# 以彩色圖片的方式載入
img1 = cv2.imread(img_path, cv2.IMREAD_COLOR)

# 以灰階圖片的方式載入
img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
 # 顯示彩圖
cv2.imshow('bgr', img1)
# 顯示灰圖
cv2.imshow('gray', img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()