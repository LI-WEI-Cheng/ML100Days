# ------------------------------------------------------------------------------------------------------------- #
# 在作邊緣偵測時，通常會調整模糊參數(cv2.GaussianBlur)或邊緣檢測參數(cv2.Canny)來達到想要的結果
#
# 步驟大約分成
#
# 影像轉灰階： cv2.cvtColor
#
# 影像去雜訊： cv2.GaussianBlur
# cv2.GaussianBlur第二個參數是指定Gaussian kernel size，本範例使用5×5
#
# 邊緣偵測： cv2.Canny
# 採用雙門檻值
# 第二個參數是指定門檻值 threshold1 – first threshold for the hysteresis procedure.
# 第二個參數是指定門檻值 threshold2 – second threshold for the hysteresis procedure.
# ------------------------------------------------------------------------------------------------------------- #
# Filter 的基本操作
# Filter 又稱做 kernel，概念上是一個固定大小的矩陣，掃過圖片經過計算後取得新的圖片矩陣
# 假設以一個 3*3 的 filter 操作，每次運算都會把 9 個 pixel 的值相乘相加得到一個值，可以把這個操作想像為特徵提取
# I * K = I*K
# (7*7) * (3*3) --> (5*5) : 因為每 9 個 pixel 的值相乘相加得到一個值
# ------------------------------------------------------------------------------------------------------------- #
# Filter 操作 - Cross-Correlation vs Convolution:
# 把 Filter 掃過圖片時我們要做兩個矩陣的運算，根據計算順序的不同主要分為 Cross-Correlation 跟 Convolution
# Cross-Correlation : f x g (由左到右，由上到下)
# --> 卷積中的濾波器不翻轉
# Convolution : g * f (由右到左，由下到上)
# --> 濾波器g首先翻轉，然後沿著橫座標移動。計算兩者相交的面積，就是卷積值。
# ------------------------------------------------------------------------------------------------------------- #
# 單個通道的卷積在深度學習中，卷積是元素對元素的加法和乘法。對於具有一個通道的影象，卷積如上圖所示。
# 在這裡的濾波器是一個3x3的矩陣。
# 濾波器滑過輸入，在每個位置完成一次卷積，每個滑動位置得到一個數字。
# 對於5*5的矩陣，最終輸出仍然是一個3x3的矩陣。(就沒有長、寬各減2)
# ------------------------------------------------------------------------------------------------------------- #
# https://www.mdeditor.tw/pl/2UEO/zh-tw
# 在很多應用中，我們需要處理多通道圖片。最典型的例子就是RGB影象，不同的通道強調原始影象的不同方面。
# 另一個多通道資料的例子是CNN中的層。卷積網路層通常由多個通道組成（通常為數百個通道）。
# 每個通道描述前一層的不同方面。我們如何在不同深度的層之間進行轉換？如何將深度為n的層轉換為深度為m下一層？
# 在描述這個過程之前，我們先介紹一些術語：layers（層）、channels（通道）、feature maps（特徵圖），filters（濾波器），kernels（卷積核）。
# 從層次結構的角度來看，層和濾波器的概念處於同一水平，而通道和卷積核在下一級結構中。通道和特徵圖是同一個事情，一層可以有多個通道（或者說特徵圖）。
# 如果輸入的是一個RGB影象，那麼就會有3個通道。“channel”通常被用來描述“layer”的結構。相似的，“kernel”是被用來描述“filter”的結構。
# ------------------------------------------------------------------------------------------------------------- #
# Filter 操作 - Padding :
# 根據前面介紹的操作都是把 n 個值經過計算得到 1 個值，這樣圖片會變小。
# 所以我們通常會在圖片周圍加上額外的值，確保運算完之後跟原圖大小一樣，周圍填甚麼值會影響最後結果
#
# 這邊根據不同情況常見的操作有以下幾種：
# 補零
# 補鄰近 pixel 值
# 補整張圖片 pixel 值的 mean
# 鏡射
# ------------------------------------------------------------------------------------------------------------- #
# Convolution 與 Cross-Correlation 基本上只有順序不一樣
# 直接對圖片做 Convolution 會使得最後圖片變小，所以通常會加上 padding
# ------------------------------------------------------------------------------------------------------------- #
import cv2
import numpy as np
from PIL import Image, ImageFilter

picture = cv2.imread('lena.jpg',cv2.IMREAD_COLOR)
picture_gray = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
picture_for_blur = picture.copy()
# ------------------------------------------------------------------------------------------------------------- #
# Filter --> 模糊 Gaussian Blur :
# 影像模糊從另一個觀點解釋可以說是邊緣不明顯，而邊緣的物理意義可以簡單代表兩側的顏色差別大
# 因此圖片要變模糊主要是物體邊界會受影響，我們也可以把模糊想像成是要讓顏色的差異變小
# 最簡單的概念是只要把周遭的 pixel 全部平均就好，但比較常使用的是 Gaussian Filter，認為中心點的資訊還是最重要的
# 有時候也會有去雜訊的效果
# cv2.GaussianBlur(picture,filter size大小,x軸的標準差，設0會根據filter size 來自己計算)
picture_blur = cv2.GaussianBlur(picture,(3,3),0)
picture_blur = cv2.GaussianBlur(picture_blur,(3,3),0)
picture_blur1 = cv2.GaussianBlur(picture_blur,(3,3),0)
picture_blur2 = cv2.GaussianBlur(picture_blur1,(3,3),0)
picture_blur2 = cv2.GaussianBlur(picture_blur2,(3,3),0)
cv2.imshow('BLUR', picture_blur)
picture_blur = np.hstack((picture, picture_blur1,picture_blur2))
cv2.imshow('BLUR', picture_blur)
cv2.waitKey(0)
cv2.destroyAllWindows()
# --------------------------------------------------------------- #
# # 法(二) : 內建 PIL
# new_img = picture_for_blur.filter(ImageFilter.BLUR)
# new_img.show()
#

# ------------------------------------------------------------------------------------------------------------- #
# 我打的是圖片的權重
# 作業解答則是 求導的階數
# ------------------------------------------------------------------------------------------------------------- #
# Filter --> 邊緣偵測 Sobel :
# 如同我們前面說的，邊緣的特性是兩側的顏色差別很大
# 跟模糊不一樣的地方是邊緣檢測是要加強邊緣特性
# 通常會用 Gray Scale 圖來做邊緣檢測
# 基本的邊緣檢測：Sobel filter
# 尋找邊緣正確的說法應該是計算圖片的導數 (derivatives)
# 我們可以透過 Sobel 個別拿到 x 與 y 方向的邊緣後再結合
# ------------------------------------------------------------- #
# Sobel算子依然是一種過濾器，只是其是帶有方向的。
# 前四個是：
# 第一個參數是需要處理的圖像
# 第二個參數是圖像的深度，-1表示採用的是與原圖像相同的深度。目標圖像的深度必須大於等於原圖像的深度；
# ksize是Sobel算子的大小，必須為1、3、5、7。
# dx和dy表示的是求導的階數，0表示這個方向上沒有求導，一般為0、1、2。
# ------------------------------------------------------------- #
# cv2.Sobel(picture_gray,希望運算過程中使用int16,dx : 代表求倒數次數,dy : 0代表不用求導 ,ksize=3)
# Sobel函數求完導數後會有負值，還有會大於255的值。而原圖像是uint8，即8位無符號數，所以Sobel建立的圖像位數不夠，會有截斷。
# 因此要使用16位有符號的數據類型，即cv2.CV_16S。
#　Sobel 如果在 uint8 的情況下做會 overflow 的狀況
picture_x = cv2.Sobel(picture_gray,cv2.CV_16S,dx = 1,dy=0,ksize=3)
picture_y = cv2.Sobel(picture_gray,cv2.CV_16S,dx = 0,dy=1,ksize=3)
# ------------------------------------------------------------- #
# 個別計算完 xy 方向的邊緣之後再接著處理
# 處理計算完的負數部份，做正規化變成非負整數再改為 uint8
# 合併 xy 邊緣的圖
picture_x = cv2.convertScaleAbs(picture_x) # 正規化
picture_y = cv2.convertScaleAbs(picture_y)
# cv2.addWeighted(picture_x,第一章圖的權重,picture_y,第二章圖的權重,家道最後結果的值)
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
# ------------------------------------------------------------------------------------------------------------- #
# HW : 解答
img = cv2.imread('lena.jpg')
# 轉為灰階圖片
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 對 x 方向以包含負數的資料格式 (cv2.CV_16S) 進行 Sobel 邊緣檢測
img_sobel_x = cv2.Sobel(img_grey, cv2.CV_16S, dx=1, dy=0, ksize=3)
# 對 x 方向依照比例縮放到所有數值都是非負整數
img_sobel_x = cv2.convertScaleAbs(img_sobel_x)

# 對 x 方向直接以非負整數的資料格式 (uint8) 進行 Sobel 邊緣檢測
img_sobel_x_uint8 = cv2.Sobel(img_grey, -1 , dx=1, dy=0, ksize=3)

#　組合 + 顯示圖片
img_show = np.hstack((img_grey, img_sobel_x, img_sobel_x_uint8))
# 比較 Sobel 邊緣檢測的過程中針對負數操作的不同產生的差異
cv2.imshow('Edge Detection', img_show)
cv2.waitKey(0)

cv2.destroyAllWindows()

img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 求一次導數取得邊緣檢測結果
img_sobel_x = cv2.Sobel(img_grey, cv2.CV_16S, dx=1, dy=0, ksize=3)
img_sobel_x = cv2.convertScaleAbs(img_sobel_x)

# 求二次導數取得邊緣檢測結果
img_sobel_xx = cv2.Sobel(img_grey, cv2.CV_16S, dx=2, dy=0, ksize=3)
img_sobel_xx = cv2.convertScaleAbs(img_sobel_xx)

#　組合 + 顯示圖片
img_show = np.hstack((img_grey, img_sobel_x, img_sobel_xx))
while True:
    cv2.imshow('Edge Detection', img_show)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
        break
# ------------------------------------------------------------------------------------------------------------- #

picture_x123 = cv2.Sobel(picture_blur1,cv2.CV_16S,dx = 1,dy=0,ksize=3)
picture_y123 = cv2.Sobel(picture_blur1,cv2.CV_16S,dx = 0,dy=1,ksize=3)
picture_x123 = cv2.convertScaleAbs(picture_x123)
picture_y123 = cv2.convertScaleAbs(picture_y123)
picture_combine = cv2.addWeighted(picture_x123,0.5,picture_y123,0.5,0)
cv2.imshow('Combine', picture_combine)
cv2.waitKey(0)
cv2.destroyAllWindows()























