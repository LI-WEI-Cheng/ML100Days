# SIFT特徵應用
# 當我們取得特徵之後，就代表我們已經有能力去識別關鍵點的特殊性
#
# 在這之後可以接到許多電腦視覺的任務
# 配對：判斷兩張圖片上相同物體的位置
# 辨識：判斷兩張圖片上是否有相同物體
# 全景圖：尋找兩張圖片的相同視角，再經過轉換合成全景圖
# ...
# 廣泛的說，SIFT 只是其中一種抽取特徵的方式，這邊會延續上一章節以 SIFT 為例介紹配對的應用。
# ---------------------------------------------------------------------------------------- #
import cv2
import numpy as np
img = cv2.imread('box.png')
picture = cv2.imread('box_in_scene.png')
img_gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
picture_gray= cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
SIFT_detector = cv2.xfeatures2d.SIFT_create()
keypoints1 = SIFT_detector.detect(img_gray , None)
keypoints2 = SIFT_detector.detect(picture_gray , None)
# ---------------------------------------------------------------------------------------- #
points123 = cv2.KeyPoint_convert(keypoints1)  #將KeyPoint格式資料中的xy座標提取出來。
points456 = cv2.KeyPoint_convert(keypoints1)  #將KeyPoint格式資料中的xy座標提取出來。
print(points123)
print(points456)
# ---------------------------------------------------------------------------------------- #
# 簡單暴力的配對方法是逐一針對 query image 的關鍵點
# 對每個 train image 的關鍵點計算 L2 距離
# 取得距離最小的配對
# 取得 k 個最適合的配對
# 這邊為了確保配對的合適性
# 可以先在計算時取得 k 個配對，再根據距離去過濾不適合的配對
# ---------------------------------------------------------------------------------------- #
# 取得"keypoint"同時"計算128維度的敘述子向量"
# 那個NONE是mask參數，如果有設定則可以針對部分圖片計算SIFT
# 第一個是query image
kp1, query1 = SIFT_detector.detectAndCompute(img_gray, None)
# 第二個是train image
kp2, query2 = SIFT_detector.detectAndCompute(picture_gray, None)
print('----------------------------------')
print(kp1)
print('----------------------------------')
print(query1)
print('----------------------------------')
print(kp2)
print('----------------------------------')
print(query2)
# ---------------------------------------------------------------------------------------- #
# SIFT 特徵 - 尺度不變性
# 配對會從兩張圖片中的關鍵點中，透過計算其特徵空間上的距離，若小於一個設定的閥值就視為是相同的特徵
# 在 SIFT 特徵的配對任務中，通常會使用 L2 norm 的方式計算，兩個 128 維向量根據上面公式計算可以得到一個距離
# 建立 Brute-Force Matching 物件
# 用來宣告要使用L2 norm計算距離
bf = cv2.BFMatcher(cv2.NORM_L2)
# ---------------------------------------------------------------------------------------- #
# 我們可以尋找 k=2 個最好的 match 方式
# 以 knn 方式暴力比對特徵
matches = bf.knnMatch( query1,  query2, k=2)
# ---------------------------------------------------------------------------------------- #
# 透過 ratio test 的方式來過濾一些不適當的配對
# 因為有時候 query 的關鍵點並不會出現在 train image
# 透過 D.Lowe ratio test 排除不適合的配對
# 0.75是ratio test設定比值為0.75
candidate = []
for m, n in matches:
    if m.distance < 0.75*n.distance:
        candidate.append([m])


# 顯示配對結果
# cv2.drawMatchesKnn(img_gray, kp1, picture_gray, kp2, candidate, 不給None會有錯誤, flags=2代表沒有配對成功的點不會被畫出來)
img_show = cv2.drawMatchesKnn(img_gray, kp1, picture_gray, kp2, candidate, None, flags=2)

# 顯示圖片
while True:
    cv2.imshow('matches', img_show)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
        break
# ------------------------------------------------------------------------------------------------------- #
# 許多機器學習的任務一開始都要先經過抽取特徵的步驟
# 諸如 SIFT 等傳統電腦視覺的特徵只是其中一種方式
# 而近期非常熱門的深度學習則是另外一種抽取特徵的方式
# 後面再根據任務類型選擇要對特徵做甚麼處理
# 所以傳統特徵跟 model 抽的特徵，使用上是差不多的
# e.g. 分類任務，我們就把抽完的特徵當作 input 輸入分類器
# ------------------------------------------------------------------------------------------------------- #
# SIFT 雖然可以做機器學習任務，但是實作上存在一些問題
# 因為演算法的關係，不保證所有圖片都會產生一樣維度的特徵
# 一般機器學習任務的 input 都要是同樣的維度
# 因此 SIFT 特徵必須做前處理
# ------------------------------------------------------------------------------------------------------- #
# SIFT 在應用上的問題 (optional)
# SIFT 每一個特徵點的維度其實是一樣的，但每張圖片產生的特徵點個數不同，才會導致圖片的特徵維度不同
# 其中一種作法是做 Clustering，每一張圖片都取 n 個特徵點來固定圖片的特徵維度
# 缺點：
# 如果圖片太簡單導致部份圖片特徵太少就會失效，所以類似 MNIST 或是 CIFAR 等簡單的資料集就不太適合
# ------------------------------------------------------------------------------------------------------- #


























































