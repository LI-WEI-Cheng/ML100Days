# ------------------------------------------------------------------------------------------------------------------- #
# 仿射變換 Affine Transformation --->  仿射變換是線性變換後再做平移的結果
# 仿射不一定會保持物體的面積與大小，但是具有以下兩種特性
# 共線不變性：平行的線經過轉換後還是會平行
# 比例不變性：兩點的中點經過轉換還會是中點
# ---------------------------------------------------------------- #
# 常見的線性變換:
# 平移 (Translation)
# 旋轉 (Rotation)
# 鏡射 (Reflection)
# 伸縮 (Stretching/ Squeezing)
# 切變 (Shearing)
# ---------------------------------------------------------------- #
# 平移 : 图像的平移，沿着x方向tx距离，y方向ty距离，那么需要构造移动矩阵：
# [1 0 tx ; 0 1 ty]
# ---------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
# 解答方法
import cv2
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

picture = cv2.imread('lena.jpg',cv2.IMREAD_COLOR)
rows, cols = picture.shape[:2]
print(rows)
print(cols)
# 取得旋轉矩陣
# getRotationMatrix2D(center, angle, scale)
M_rotate = cv2.getRotationMatrix2D((cols//2, rows//2), 45, 0.5)
print('Rotation Matrix')
print(M_rotate)
print()

# 取得平移矩陣
M_translate = np.array([[1, 0, 100], [0, 1, -50]], dtype=np.float32)
print('Translation Matrix')
print(M_translate)

# 旋轉
img_rotate = cv2.warpAffine(picture, M_rotate, (cols, rows))

# 平移
img_rotate_trans = cv2.warpAffine(img_rotate, M_translate, (cols, rows))

# 組合 + 顯示圖片
img_show_rotate_trans = np.hstack((picture, img_rotate, img_rotate_trans))
while True:
    cv2.imshow('Rotate 45, scale 0.5, Translate x+100, y-50', img_show_rotate_trans)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
        break
# ------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #

picture = cv2.imread('lena.jpg',cv2.IMREAD_COLOR)
# ------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
# 解答方法
rows, cols = picture.shape[:2]

# 取得旋轉矩陣
# getRotationMatrix2D(center, angle, scale)
M_rotate = cv2.getRotationMatrix2D((cols//2, rows//2), 45, 0.5)
print('Rotation Matrix')
print(M_rotate)
print()

# 取得平移矩陣
M_translate = np.array([[1, 0, 100], [0, 1, -50]], dtype=np.float32)
print('Translation Matrix')
print(M_translate)

# 旋轉
img_rotate = cv2.warpAffine(picture, M_rotate, (cols, rows))

# 平移
img_rotate_trans = cv2.warpAffine(img_rotate, M_translate, (cols, rows))

# 組合 + 顯示圖片
img_show_rotate_trans = np.hstack((picture, img_rotate, img_rotate_trans))
while True:
    cv2.imshow('Rotate 45, scale 0.5, Translate x+100, y-50', img_show_rotate_trans)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
        break
# ------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
# 法二(但有點不太一樣)
def rotate_bound1(image, angle):
    (h, w) = image.shape[:2]  # 返回(高,寬,色彩通道数),此取前兩個值
    # 抓取旋转矩阵(应用角度的负值顺时针旋转)
    # cv2.getRotationMatrix2D(旋轉中心點,旋轉角度(逆為正),比例大小)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 0.5)
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
picture = rotate_bound1(picture, -45)

def translate(image, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifte = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifte
shifte1 = translate(picture, 100,-50)
cv2.imshow('Translation', shifte1)
cv2.waitKey(0)
cv2.destroyAllWindows()
# ------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
# 作業的解答 : 先定義轉換前後的三點，利用三對點找轉至矩陣，由矩陣來旋轉，再利用for迴圈將個圖個點印出來
import cv2
import time
import numpy as np
img = cv2.imread('lena.jpg',cv2.IMREAD_COLOR)
# 給定兩兩一對，共三對的點
# 這邊我們先用手動設定三對點，一般情況下會有點的資料或是透過介面手動標記三個點
(rows, cols) = img.shape[:2]
pt1 = np.array([[50,50], [300,100], [200,300]], dtype=np.float32)
pt2 = np.array([[80,80], [330,150], [300,300]], dtype=np.float32)
# 取得 affine 矩陣並做 affine 操作
M_affine = cv2.getAffineTransform(pt1, pt2)
img_affine = cv2.warpAffine(img, M_affine, (cols, rows))

# 在圖片上標記點 :
img_copy = img.copy()
# enumerate在字典上是枚舉、列舉的意思
# enumerate參數為可遍歷/可叠代的對象(如列表、字符串)
# enumerate多用於在for循環中得到計數，利用它可以同時獲得索引和值，即需要index和value值的時候可以使用enumerate
# enumerate()返回的是一個enumerate對象
for idx, pts in enumerate(pt1):
    pts = tuple(map(int, pts))
    print(pts)
    cv2.circle(img_copy, pts, 3,(255, 255, 0), -1)
    cv2.putText(img_copy, str(idx), (pts[0]+5, pts[1]+5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 1)

for idx, pts in enumerate(pt2):
    pts = tuple(map(int, pts))
    print(pts)
    cv2.circle(img_affine, pts, 3,(255, 255, 0), -1)
    cv2.putText(img_affine, str(idx), (pts[0]+5, pts[1]+5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 1)

# 組合 + 顯示圖片
img_show_affine = np.hstack((img_copy, img_affine))
cv2.imshow('affine transformation', img_show_affine)
cv2.waitKey(0)
cv2.destroyAllWindows()
















































































