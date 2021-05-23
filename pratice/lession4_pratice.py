# Image augmentation 兩個主要的功能包含『彌補資料不足』以及『避免Overfitting』 :
# 該如何自己創造新的資料呢？最簡單的方式就是透過Image augmentation，我們藉由旋轉、裁切、增加噪點、白化等技術，如此一來，我們就硬生生地增加了許多的資料。
# 訓練一個分類器時，大家應該很容易遇到Overfitting的狀況，也就是對Training Data過於完美的擬合，此時，透過適當的圖像增強，也能降低Overfitting 的可能性。
# Image Augmentation 是常見的影像前處理，然而也要避免一些錯誤的使用情境:
# 如訓練數字模型時使用垂直翻轉，這樣會造成6、9之間的訓練問題，
# 如輸入影像為小尺寸(ex. 32*32)，結果隨機裁切16個像素，如此幾乎所有的內容都被裁切導致模型無法學到有用資訊。
# --------------------------------------------------------------------------------------------------------------------- #
# 水平與垂直翻轉 (Flip) :
# xy 軸順序顛倒
# --------------------------------------------- #
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
# --------------------------------------------- #
picture = cv2.imread('unnamed.jpg',cv2.IMREAD_COLOR)
# --------------------------------------------- #
# 表示圖片channel方法 : img[:,:,1] or img[...,1]
# ::-1代表從 end 走到 start (倒序)
# [::-1,:,:] 代表對y軸倒序(高)
# [:,::-1,:] 代表對x軸倒序(寬)
# [:,:,::-1] 代表對channel倒序
filp_vertical = picture[::-1,:,:]
filp_horizontal = picture[:,::-1,:]
filp_horizontal_vertical = filp_horizontal[::-1,:,:]
# --------------------------------------------- #
img_combine_first = np.hstack((picture,filp_horizontal ))
img_combine_second = np.hstack((filp_vertical,filp_horizontal_vertical))
img_combine_first = cv2.cvtColor(img_combine_first, cv2.COLOR_BGR2RGB)
img_combine_second = cv2.cvtColor(img_combine_second, cv2.COLOR_BGR2RGB)
plt.figure()
plt.subplot(2,1,1)
plt.imshow(img_combine_first)
plt.axis('off')
plt.subplot(2,1,2)
plt.imshow(img_combine_second)
plt.axis('off')
plt.show()
# --------------------------------------------- #
img_combine_all = np.vstack((img_combine_first,img_combine_second))
img_combine_all = cv2.cvtColor(img_combine_all, cv2.COLOR_BGR2RGB)
cv2.imshow('Combine', img_combine_all)
cv2.waitKey(0)
cv2.destroyAllWindows()
# --------------------------------------------- #
# --------------------------------------------------------------------------------------------------------------------- #
# 縮放操作 (Scale) - OpenCV
# 因為縮小跟放大要參考周圍的 pixel 值
# 經過統計與運算去減少 / 生成新的 pixel 值
# interpolation	說明 : [ 放操作的方式 (interpolation) 會影響處理的速度與圖片品質 ]
# INTER_NEAREST	-> 最近鄰插值
# INTER_LINEAR	-> 雙線性插值（預設）
# INTER_AREA	-> 使用像素區域關係進行重採樣。它可能是圖像抽取的首選方法，因為它會產生無雲紋理(波紋)的結果。 但是當圖像縮放時，它類似於INTER_NEAREST方法。
# INTER_CUBIC	-> 4x4像素鄰域的雙三次插值
# INTER_LANCZOS4-> 8x8像素鄰域的Lanczos插值
# --------------------------------------------- #
# 檢查原始圖片大小-openCV
sp = picture.shape  # [高|宽|像素值由三种原色构成]
print(sp)
# --------------------------------------------- #
# 縮放操作 (Scale) - 以不同內插法做縮放
# 有兩中操作方式
# cv2.resize(圖,直接定義輸出的大小,哪種運算方法)
# cv2.resize(圖,None,fx(輸出圖片的比列是原比例的多少倍),fy(輸出圖片的比列是原比例的多少倍),哪種運算方法)
picture_scale_change1 = cv2.resize(picture,(512,512),interpolation=cv2.INTER_NEAREST)
picture_scale_change2 = cv2.resize(picture,(512,512),interpolation=cv2.INTER_LINEAR)
picture_scale_change3 = cv2.resize(picture,(512,512),interpolation=cv2.INTER_AREA)
picture_scale_change4 = cv2.resize(picture,(512,512),interpolation=cv2.INTER_CUBIC)
picture_scale_change5 = cv2.resize(picture,(512,512),interpolation=cv2.INTER_LANCZOS4)
img1_rgb = cv2.cvtColor(picture_scale_change1, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(picture_scale_change2, cv2.COLOR_BGR2RGB)
img3_rgb = cv2.cvtColor(picture_scale_change3, cv2.COLOR_BGR2RGB)
img4_rgb = cv2.cvtColor(picture_scale_change4, cv2.COLOR_BGR2RGB)
img5_rgb = cv2.cvtColor(picture_scale_change5, cv2.COLOR_BGR2RGB)
titles = ['Original Image', 'INTER_NEAREST', 'INTER_LINEAR', 'INTER_AREA', 'INTER_CUBIC', 'INTER_LANCZOS4']
images = [picture, img1_rgb, img2_rgb, img3_rgb, img4_rgb, img5_rgb]
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
# --------------------------------------------- #
# 如果是要縮小圖片的話，通常 INTER_AREA 使用效果較佳。
# 如果是要放大圖片的話，通常 INTER_CUBIC 使用效果較佳，次等則是 INTER_LINEAR。
# 如果要追求速度的話，通常使用 INTER_NEAREST。
# --------------------------------------------- #
# INTER_AREA vs. INTER_CUBIC
titles123 = [ 'INTER_LINEAR', 'INTER_CUBIC', ]
images123 = [ img3_rgb, img4_rgb]
for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.imshow(images123[i])
    plt.title(titles123[i])
    plt.xticks([]), plt.yticks([])
plt.show()
# 可以明顯看到 INTER_AREA
# 較模糊且有鋸齒邊緣 (左)
# -------------------------------------------------------------------------------------------#
# 法(二) 等比例放大、縮小
# 將圖片縮小成原本的 20%

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
# --------------------------------------------- #
# --------------------------------------------------------------------------------------------------------------------- #
# 平移操作 (Translation Transformation) :
# 所謂的平移操作的意義為物體往某個向量方向移動，但是其形狀，結構與視角都不發生改變
# 方法一：手動做 xy 軸的四則運算 取得移動後的位置 (慢)
# 方法二：以矩陣運算方式操作 (快)
#   圖片與 Transformation Matrix 相乘
#   一次性操作就可以得到平移後的值，來得到新的pixel位置
# --------------------------------------------- #
# 平移操作的向量移動方式 : (利用這兩個方程式轉成3x3的矩陣)
# x' = ax+cy+e
# y' = bx+dy+f
# x, y 軸的移動不會考慮 y, x 軸的值：c = b = 0
# x, y 軸的移動不會做 scale：a = d = 1
# e = x 軸移動多少 pixel
# f = y 軸移動多少 pixel
# --------------------------------------------- #
# OpenCV :
# 在openCV中，當我們給定不一樣的 Matrix 就可以做不一樣的 Transformation
# --------------------------------------------- #
# 用numPy的陣列作為構建基礎，專門用來處理矩陣，它的運算效率比列表更高效。
# np.array：一種多維陣列物件
# np.array(a,b) : a * b 矩陣
# 需要知道你所處理的資料的大致類型是浮點數、複數、整數、布林值、字串，還是普通的 python 對象。當需要控制資料在記憶體和磁片中的存儲方式時，就得瞭解如何控制存儲類型。
# 所以，dtype（資料類型）是一個特殊的物件，它含有 ndarray 將一塊記憶體解釋為特定資料類型所需的資訊。
# 故這邊直接使用的是np.float32
# 定義矩阵 向右平移10个像素， 向下平移50个像素
M = np.float32([[1, 0, 10], [0, 1, 50]]) #透過 np.array 產生平移矩陣
# 用OpenCV進行2D變換
# picture.shape[0-2] : 寬、高、channel
shifted = cv2.warpAffine(picture, M, (picture.shape[1], picture.shape[0]))
cv2.imshow('Translation', shifted)
cv2.waitKey(0)
cv2.destroyAllWindows()
# --------------------------------------------- #
# 封裝看看:(def)
def translate(image, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifte = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifte
shifte1 = translate(picture, 10, 30)
cv2.imshow('Translation', shifte1)
cv2.waitKey(0)
cv2.destroyAllWindows()
# --------------------------------------------- #






# --------------------------------------------------------------------------------------------------------------------- #


































