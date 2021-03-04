# 改變 color space 來調整、操作飽和度 :
# ---------------------------------------------------　#
import cv2
# 首先，RGB是依序稱為0~2的channel : 若單純做飽和度在HSV中要抓第1層channel的row*column(一個矩陣) 叫img[:,:,1] or img[...,1]
# 若選其他channel則代表的會是 H或V的改變~
# 所以做飽和度調整時，可以先將圖檔轉乘HSV後取其中的channel
# 因為 uint8 (可儲存範圍 0~255)，所以 OpenCV 會以 0~255 的值表示
# 1.抓圖 --> 2.轉HSV --> 3. 將要操作的那層飽和度轉為浮點數
picture = cv2.imread('lena.jpg',cv2.IMREAD_COLOR)
img1 = cv2.cvtColor(picture, cv2.COLOR_BGR2HSV)
img1 = img1.astype('float32')  # 操作飽和度時，需要轉換成百分比(小數點)
# 飽和度是0-225 所以要調降2成飽和度的話，可以先除225再減0.2，但不能減0.2*225
# 兩種運算矩陣不同
# print(img1[...,1]/255) #: 是一個矩陣
# a = img1[...,1]/255-0.2
# b = img1[...,1]-0.2*255
# print("--------------------------------")
# print(a)
# print("--------------------------------")
# print(b)
# 為了減少第一層(Green)的兩成飽和度，所以
img1[...,1] = img1[...,1]/255-0.2
# 做邊緣條件判斷 確保所有值在合理範圍(0-1之間)
img1[img1[...,1] < 0] = 0
img1[...,1] = img1[...,1]*255
# 轉換成0到255(uint格式)
img1 = img1.astype('uint8')
# 轉到RBG格式
img_down = cv2.cvtColor(img1, cv2.COLOR_HLS2BGR)
# 顯示圖片 (imshow要是0-255)
cv2.imshow('DOWN SATURATION', img_down)
cv2.waitKey(0)
cv2.destroyAllWindows()
# ---------------------------------------------------　#
# 直方圖均勻化(Histogram Equalization) :
# Histogram 最理想分佈狀況就是平均分佈
# 來增加許多圖像的全局對比度，尤其是當圖像的有用數據的對比度相當接近的時候
# 對於背景和前景都太亮或者太暗的圖像非常有用
# 主要優勢是它是一個相當直觀的技術並且是可逆操作
# 缺點是它對處理的數據不加選擇，它可能會增加背景雜訊的對比度並且降低有用訊號的對比度
# 主要是處理灰圖，如果處理 RGB 圖通常是會轉到 HSV 再對明亮度做直方圖均衡，不過我們也可以個別對 RGB 的 3 個 channel 做直方圖均衡
# ---------------------------------------------------　#
import numpy as np
# 一個由多維陣列物件和用於處理陣列的例程集合組
# 執行以下操作：
# 1.陣列的算數和邏輯運算。
# 2.傅立葉變換和用於圖形操作的例程.
# 3.與線性代數有關的操作。 NumPy 擁有線性代數和隨機數生成的內建函式。
# ---------------------------------------------------　#
# 轉為灰階圖片
img_gray = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
# 灰階圖片直方圖均衡
img_gray_equal = cv2.equalizeHist(img_gray)
# np.hstack水平將矩陣堆疊 # np.tstack垂直將矩陣堆疊
img_gray_equalHist = np.hstack((img_gray, img_gray_equal))
cv2.imshow('gray equal histogram', img_gray_equalHist)
cv2.waitKey(0)
cv2.destroyAllWindows()
# ---------------------------------------------------　#
# 轉到 HSV 再對明亮度做直方圖均衡
# 透過數學公式將值分散到 0~255 區間
picture1 = cv2.imread('lena.jpg',cv2.IMREAD_COLOR)
img_HSV_HE = cv2.cvtColor(picture1, cv2.COLOR_BGR2HSV)
img_HSV_HE_equal = cv2.equalizeHist(img_HSV_HE[...,2])
img_add_HSV = np.hstack((img_gray, img_gray_equal,img_HSV_HE_equal))
cv2.imshow('CHANGE V', img_add_HSV )
cv2.waitKey(0)
cv2.destroyAllWindows()
img_HSV_HE[...,2] =  img_HSV_HE_equal
img_final = cv2.cvtColor(img_HSV_HE , cv2.COLOR_HSV2BGR)
cv2.imshow('CHANGE final-1', img_final )
cv2.waitKey(0)
cv2.destroyAllWindows()
# ---------------------------------------------------　#
# 個別對 RGB 的 3 個 channel 做直方圖均衡
img_rgb = cv2.imread('lena.jpg',cv2.IMREAD_COLOR)
# img_RGB_R = cv2.equalizeHist(img_rgb[...,0])
# img_RGB_G = cv2.equalizeHist(img_rgb[...,1])
# img_RGB_B = cv2.equalizeHist(img_rgb[...,2])
# # 由HSV改後再轉RGB才要個別在定義一次
# img_rgb[...,0] = img_RGB_R
# img_rgb[...,1] = img_RGB_G
# img_rgb[...,2] = img_RGB_B
img_rgb[...,0] = cv2.equalizeHist(img_rgb[...,0])
img_rgb[...,1] = cv2.equalizeHist(img_rgb[...,1])
img_rgb[...,2] = cv2.equalizeHist(img_rgb[...,2])
cv2.imshow('CHANGE final-2',img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
img_all_change_combine = np.hstack((picture,img_final,img_rgb))
cv2.imshow('CHANGE final-3',img_all_change_combine)
cv2.waitKey(0)
cv2.destroyAllWindows()
# ---------------------------------------------------　#
# 手動調整對比度跟明亮度:
# g(x)->調整過後的pixel值 = apha*f(x)->原來圖片的pixel值+beta
# apha調整對比度(range : 1.0-3.0)
# beta調整明亮度(range : 0-100)
# 可以透過for迴圈逐一對每一個pixel 調整
# 也可以透過openCV內建: img = cv2.convertScaleAbs(img, alphal = alpha ,beta = beta)
# 因為圖片的值限制在0-255之間，所以參數值不能太大:
# alpha: 控制對比度 (1.0~3.0)
# beta: 控制明亮度 (0~255)
#下列以openCV內建的函式調整
add_contrast = cv2.convertScaleAbs(picture1, alpha=2.0, beta=0)
add_lighness = cv2.convertScaleAbs(picture1, alpha=1.0, beta=50)
img_contrast_light = np.hstack((picture1, add_contrast, add_lighness))
# 比較不同程度的對比 / 明亮
cv2.imshow('adjust contrast and brighness', img_contrast_light)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 下列透過foe迴圈來對逐一每個pixel調整
alpha=2.0
beta=0
print(picture1.shape[0])
for b in range (picture1.shape[0]):
    for g in range(picture1.shape[1]):
        for r in range  (picture1.shape[2]):
            # 截取的意思，超出的部分就把它強置為邊界部分: 限制範圍在0-255
            picture1[b,g,r] = np.clip(alpha*picture1[b,g,r]+beta,0,255)
cv2.imshow('adjust contrast and brighness by for loop', picture1)
cv2.waitKey(0)
cv2.destroyAllWindows()
# ---------------------------------------------------　#
# 最後，為了可以一次顯示圖片，要使用" import matplotlib.pyplot as plt "
# 但是matplotlib的輸出順序是 : RGB
# 而OPENCV的順序是BGR
# 所以需要先做轉換!
# # BGR转RGB，方法1
# img_rgb1 = cv2.merge([R, G, B])
#
# # BGR转RGB，方法2
# img_rgb2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
# # BGR转RGB，方法3
# img_rgb3 = img[:, :, ::-1] --> [長,寬,channel] -->[:,:,::-1(是順序調換的意思)]
# [..,-1]:則是取最後一個channel
# 再來是sublot
# plt.subplot(3,3,1), plt.imshow(img) 代表[A x x ; x x x ; x x x]中在A的位置秀出img
# 而最後還需要打 : plt.show()












