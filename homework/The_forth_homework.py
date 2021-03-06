import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
# 水平與垂直翻轉 (Flip) :
picture = cv2.imread('lena.jpg',cv2.IMREAD_COLOR)
filp_vertical = picture[::-1,:,:]
filp_horizontal = picture[:,::-1,:]
filp_horizontal_vertical = filp_horizontal[::-1,:,:]
img_combine_first = np.hstack((picture,filp_horizontal ))
img_combine_second = np.hstack((filp_vertical,filp_horizontal_vertical))
img_combine_first = cv2.cvtColor(img_combine_first, cv2.COLOR_BGR2RGB)
img_combine_second = cv2.cvtColor(img_combine_second, cv2.COLOR_BGR2RGB)
img_combine_all = np.vstack((img_combine_first,img_combine_second))
img_combine_all = cv2.cvtColor(img_combine_all, cv2.COLOR_BGR2RGB)
cv2.imshow('Combine', img_combine_all)
cv2.waitKey(0)
cv2.destroyAllWindows()
# --------------------------------------------------------------------------------------------------------------------- #
# 縮放操作 (Scale) :
# 法(一)
picture_scale_change3 = cv2.resize(picture,(256,256),interpolation=cv2.INTER_AREA)
picture_scale_change4 = cv2.resize(picture,(256,256),interpolation=cv2.INTER_CUBIC)
img3_rgb = cv2.cvtColor(picture_scale_change3, cv2.COLOR_BGR2RGB)
img4_rgb = cv2.cvtColor(picture_scale_change4, cv2.COLOR_BGR2RGB)
titles123 = [ 'INTER_LINEAR', 'INTER_CUBIC', ]
images123 = [ img3_rgb, img4_rgb]
for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.imshow(images123[i])
    plt.title(titles123[i])
    plt.xticks([]), plt.yticks([])
plt.show()
# 法(二) 等比例放大、縮小
# 將圖片縮小成原本的 20%

# 將圖片放大為"小圖片"的 8 倍大 = 原圖的 1.6 倍大

# area
start_time = time.time() # 計算時間用
img_area_scale = cv2.resize(picture, None, fx=1.6, fy=1.6,interpolation=cv2.INTER_AREA)
print('INTER_NEAREST zoom cost {}'.format(time.time() - start_time))

# cubic
start_time = time.time() # 計算時間用
img_cubic_scale = cv2.resize(picture, None, fx=1.6, fy=1.6,interpolation=cv2.INTER_CUBIC)
print('INTER_CUBIC zoom cost {}'.format(time.time() - start_time))
img5_rgb = cv2.cvtColor(img_area_scale, cv2.COLOR_BGR2RGB)
img6_rgb = cv2.cvtColor(img_cubic_scale, cv2.COLOR_BGR2RGB)
img_zoom = np.hstack((img_area_scale, img_cubic_scale))
titles123 = [ 'INTER_LINEAR', 'INTER_CUBIC', ]
images1234 = [img5_rgb,img6_rgb]
for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.imshow(images1234[i])
    plt.title(titles123[i])
    plt.xticks([]), plt.yticks([])
plt.show()

# --------------------------------------------------------------------------------------------------------------------- #
# 平移操作 (Translation Transformation) :
M = np.float32([[1, 0, 10], [0, 1, 50]])
def translate(image, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifte = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifte
shifte1 = translate(picture, 10, 30)
cv2.imshow('Translation', shifte1)
cv2.waitKey(0)
cv2.destroyAllWindows()






