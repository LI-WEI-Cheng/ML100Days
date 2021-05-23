# 適當的在圖片上做標記會讓我們更容易了解結果的正確性
#
# 偵測：標記預選框 / 偵測框等
# 遮罩：標記遮照等
# 追蹤：標記時間內移動軌跡 / 輪廓等
# 預處理：檢查經過 transformation 後的標記是否正確等
# img_rect = img.copy() 可以不斷複製相同的圖片，就不用一值重新定義!!!!!!!
# img_rect = img.copy() 可以不斷複製相同的圖片，就不用一值重新定義!!!!!!!
# img_rect = img.copy() 可以不斷複製相同的圖片，就不用一值重新定義!!!!!!!
# img_rect = img.copy() 可以不斷複製相同的圖片，就不用一值重新定義!!!!!!!
# img_rect = img.copy() 可以不斷複製相同的圖片，就不用一值重新定義!!!!!!!
# ------------------------------------------------------------ #
import cv2
import numpy as np
import matplotlib.pyplot as plt
picture = cv2.imread("lena.jpg",cv2.IMREAD_COLOR)
img_hw = picture.copy()
img_hw2 = picture.copy()
# ------------------------------------------------------------ #
# -------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------ #
# plot rectangle in the picture
img_rect = picture.copy()
# cv2.rectangle(圖片,矩形左上角座標,矩形右下角座標,BGR的顏色(線或矩形、文字的顏色),線的粗細程度)
# 我發現圖片在此的xy軸座標是往右跟往下為正
picture_rectangle1 = cv2.rectangle(img_rect,(60,40),(420,510),(0,0,255),3)
# cv2.rectangle(圖片,矩形左上角座標,矩形右下角座標,RGB的顏色,-1) : 把矩形填滿
picture_rectangle2 = cv2.rectangle(img_rect,(60,40),(420,510),(0,0,255),-1)
img_rectanlge_combine1 = np.hstack((picture_rectangle1,picture_rectangle2))
cv2.imshow('RECTANGLE1',img_rectanlge_combine1)
cv2.waitKey(0)
cv2.destroyAllWindows()
# ------------------------------------------------------------ #
# Note：OpenCV 畫圖是直接畫在圖片上面，並不是回傳一個畫好圖的圖片，所以opencv會把填滿的直接畫上去，導致兩張圖都填滿
picture_rectangle11 = cv2.rectangle(picture,(10,0),(470,560),(0,0,255),4)
picture_rectangle22 = cv2.rectangle(picture,(60,40),(420,510),(0,0,255),-1)
img_rectanlge_combine2 = np.hstack((picture_rectangle11,picture_rectangle22))
cv2.imshow('RECTANGLE1',img_rectanlge_combine2)
cv2.waitKey(0)
cv2.destroyAllWindows()
# ------------------------------------------------------------ #
# -------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------ #
# plot line in the picture.
# 通常在標記路徑 / 輪廓時會使用線
picture_line_1 = cv2.line(picture,(60,40),(420,510),(0,0,255),3)
# Lin3沒有-1的用法
# picture_line_2 = cv2.line(picture,(60,40),(420,510),(0,0,255),-1)
# img_line_combine1 = np.hstack((picture_line_1,picture_line_2))
# cv2.imshow('LINE',img_line_combine1)
cv2.imshow('LINE_1',picture_line_1)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Note：OpenCV 畫圖是直接畫在圖片上面，所以在picture的圖已經被畫上矩形了!
# ------------------------------------------------------------ #
# 所以重新輸入乾淨一張圖片
picture123 = cv2.imread("lena.jpg",cv2.IMREAD_COLOR)
picture_line_2 = cv2.line(picture123,(60,40),(420,510),(0,0,255),3)
cv2.imshow('LINE_2',picture_line_2)
cv2.waitKey(0)
cv2.destroyAllWindows()
# ------------------------------------------------------------ #
# -------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------ #
# OpenCV 畫圖 - 文字
picture123456 = cv2.imread("lena.jpg",cv2.IMREAD_COLOR)
img_word = picture.copy()
# cv2.putText(圖片,要加的文字,文字座標,字形,字體大小,BGR的顏色(線或矩形、文字的顏色),線的粗細程度)
picture_word_1 = cv2.putText(img_word,'YOU see.',(60,40),0,2,(255,255,0),3)
cv2.imshow('WORD',picture_word_1)
cv2.waitKey(0)
cv2.destroyAllWindows()
# ------------------------------------------------------------ #
# -------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------ #
# 假設我們希望先對圖片做以下幾點預處理，請印出最後結果 :
# Hint: 注意先後順序，人物原始邊框座標 (60, 40), (420, 510)
# 對明亮度做直方圖均衡處理
# 水平鏡像 + 縮放處理 (0.5 倍)
# 畫出人物矩形邊框
picture_for_example = cv2.imread("lena.jpg",cv2.IMREAD_COLOR)
img_last = picture_for_example.copy()
picture_for_example = cv2.cvtColor(picture_for_example,cv2.COLOR_BGR2HSV)
picture_for_example[...,2]=  cv2.equalizeHist(picture_for_example[...,2])
picture_for_example_change_value = cv2.cvtColor(picture_for_example, cv2.COLOR_HSV2BGR)
picture_for_example_projector = picture_for_example_change_value[:,::-1,:]
# 如果是要縮小圖片的話，通常 INTER_AREA 使用效果較佳。
picture_for_example_scale =  cv2.resize(picture_for_example_projector, None, fx=0.5, fy=0.5,interpolation=cv2.INTER_AREA)
picture_for_example_retangle = cv2.rectangle(picture_for_example_scale,(30,25),(230,250),(0,0,255),3)
img_combine_Last_1 = cv2.cvtColor(img_last, cv2.COLOR_BGR2RGB)
img_combine_Last_2 = cv2.cvtColor(picture_for_example_retangle, cv2.COLOR_BGR2RGB)
cv2.imshow('SACLE',picture_for_example_retangle)
cv2.waitKey(0)
cv2.destroyAllWindows()
# 用這個顯示出來的圖片會又便一樣大小耶!!!
plt.figure()
plt.subplot(2,1,1)
plt.imshow(img_combine_Last_1)
plt.axis('off')
plt.subplot(2,1,2)
plt.imshow(img_combine_Last_2)
plt.axis('off')
plt.show()
# -------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------ #
# 再來你會發現因為圖片縮放過了，所以所畫的框框變得不是對準人臉了，需要修正
# 在作業裡我適用try and error來做移動框架
# 這裡練習答案解法
# ------------------------------------------------------------ #
# 解法(一)
# 如果希望得知矩型邊框的位置
# 顏色的操作 (對明亮度做直方圖均衡)
# 鏡像可以透過四則運算得知
# 透過建構 transformation matrix 做縮放
# 把矩型邊框的點與 transformation matrix 相乘就會得到縮放後的位置
# 畫圖
# 得到的圖的結果正確，同時也知道新的矩型邊框座標點
# img_hw = picture.copy() --> 我在最前面定義了
# 原始 BGR 圖片轉 HSV 圖片
img_hw = cv2.cvtColor(img_hw, cv2.COLOR_BGR2HSV)
# 對明亮度做直方圖均衡 -> 對 HSV 的 V 做直方圖均衡
img_hw[..., -1] = cv2.equalizeHist(img_hw[..., -1])
# 將圖片轉回 BGR
img_hw = cv2.cvtColor(img_hw, cv2.COLOR_HSV2BGR)
# 水平鏡像
# shape : 看圖片矩陣的大小而已
h, w = img_hw.shape[:2]
print(h)
print(w)
# 圖片鏡像
img_hw = img_hw[:, ::-1, :]
# 透過四則運算計算鏡像後位置
# 由原始圖片減去框架的點，確保點的位置一樣是左上跟右下，所以交換鏡像後的 x 座標 (y 座標做水平鏡像後位置不變)
point1 = [60, 40]
point2 = [420, 510]
# 框架減去左上角點
point1[0] = w-point1[0]
# 框架減去右下角點
point2[0] = w-point2[0]
# 至此得到框架左上角點到終點(512)也就是變右上角的點，和右下角點到終點(512)的量也就是變左下角的點
print(point1)
print(point2)
# 交換鏡像後的 x 座標，所以左上角點到終點(512)和右下角點到終點(512)的量交換，所以又變回左上、右下角的點
point1[0], point2[0] = point2[0], point1[0]
print(point1)
print(point2)
# ------------------------------------------------------------ #
# 利用平移的指令將圖片縮放，並沒有任何平移
# ------------------------------------------------------------ #
# (y 座標做水平鏡像後位置不變)
fx = 0.5
fy = 0.5
resize_col = int(img_hw.shape[1]*fx)
resize_row = int(img_hw.shape[0]*fy)
# 建構 scale matrix (我想應該是一樣因為圖片有長寬channel三層，所以要另為3*3矩陣)
M_scale = np.array([[0.5, 0, 0],
                    [0, 0.5, 0]], dtype=np.float32)
img_hw = cv2.warpAffine(img_hw, M_scale, (resize_col, resize_row))
cv2.imshow('TRY',img_hw )
cv2.waitKey(0)
cv2.destroyAllWindows()
# ------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------- #
# 至此圖片變為0.5倍大小
# ------------------------------------------------------------------------------------------------------------------- #
# 下面則是利用上述平移用到的轉移矩陣來找縮放後矩陣的點
# ------------------------------------------------------------------------------------------------------------------- #
# 把左上跟右下轉為矩陣型式
bbox = np.array((point1, point2), dtype=np.float32)
print(M_scale)
print(bbox)
print('M_scale.shape={}, bbox.shape={}'.format(M_scale.shape, bbox.shape))
# 做矩陣乘法可以使用 `np.dot`, 為了做矩陣乘法, M_scale 需要做轉置之後才能相乘
homo_coor_result = np.dot(M_scale.T, bbox)
# --> 在這邊應該就只是對兩點都*0.5
# ------------------------------------------------------------ #
a = type(homo_coor_result)# 是 numoy 表示圖片格式
print(a)
# ------------------------------------------------------------ #
homo_coor_result = homo_coor_result.astype('uint8')
# uint8 : 8 位無符號整數
print(homo_coor_result)
# ------------------------------------------------------------ #
# Tuple(元組)：不能修改的List
scale_point1 = tuple(homo_coor_result[0])
scale_point2 = tuple(homo_coor_result[1])
print('origin point1={}, origin point2={}'.format(point1, point2))
print('resize point1={}, resize point2={}'.format(scale_point1, scale_point2))
# 畫圖
cv2.rectangle(img_hw, scale_point1, scale_point2, (0, 0, 255), 3)
cv2.imshow('image', img_hw)
cv2.waitKey(0)
cv2.destroyAllWindows()
print('--------------------------------------------------------------------------------------------------------------')
# -------------------------------------------------------------------------------------------------------------------- #
#  解法(二)
# 把矩型邊框用遮罩的方式呈現，使用同樣處理圖片的方式處理遮罩 最後再找遮罩的左上跟右下的點的位置
# 這邊會用到許多沒提過的東西，所以當作 optional
# 用選定的影象、圖形或物體，對待處理的影象（區域性或全部）進行遮擋來控制影象處理的區域或處理過程。
# 由於覆蓋的特定影象或物體稱為掩模(mask)，在做影象處理的時候，對影象進行遮罩的需求非常多
# 2D mask
point1 = (60, 40)
point2 = (420, 510)
# 建立掩膜(先建立一方型陣列 --> np.zeros，給定mask陣列座標 --> mask[10:170, 50:220] = 255)
# mask = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
# mask[10:170, 50:220] = 255
# np.zeros_like : 依據給定陣列的形狀和型別返回一個新的元素全部為0的陣列。
img_mask = np.zeros_like(img_hw2)
img_mask[40:510, 60:420, ...] = 255
# 原始 BGR 圖片轉 HSV 圖片
img_hw2 = cv2.cvtColor(img_hw2, cv2.COLOR_BGR2HSV)
# 對明亮度做直方圖均衡 -> 對 HSV 的 V 做直方圖均衡
img_hw2[..., -1] = cv2.equalizeHist(img_hw2[..., -1])
# 將圖片轉回 BGR
img_hw2 = cv2.cvtColor(img_hw2, cv2.COLOR_HSV2BGR)
"""
水平鏡像 + 縮放處理 (0.5 倍)
"""
# 水平鏡像 (圖片)
img_hw2 = img_hw2[:, ::-1, :]
img_mask = img_mask[:, ::-1, :]
# 縮放處理
img_hw2 = cv2.resize(img_hw2, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
img_mask = cv2.resize(img_mask, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
# ------------------------------------------------------------ #
# 取得遮照的左上跟右下的點 :
# 這邊你可以發現 transformation matrix 跟雙線性差值的結果會有一點點差異
all_h_coor, all_w_coor, all_c_coor = np.where(img_mask)
scale_point1 = (min(all_w_coor), min(all_h_coor))
scale_point2 = (max(all_w_coor), max(all_h_coor))
print('origin point1={}, origin point2={}'.format(point1, point2))
print('resize point1={}, resize point2={}'.format(scale_point1, scale_point2))
# ------------------------------------------------------------ #
# numpy.where() 有两种用法：
# 1. np.where(condition, x, y)
# 满足条件(condition)，输出x，不满足输出y。
# 2. np.where(condition)
# 只有条件 (condition)，没有x和y，则输出满足条件 (即非0) 元素的坐标 (等价于numpy.nonzero)。
# ------------------------------------------------------------ #
"""
畫出人物矩形邊框
"""
cv2.rectangle(img_hw2, scale_point1, scale_point2, (0, 0, 255), 3)
cv2.imshow('image', img_hw2)
cv2.waitKey(0)
cv2.destroyAllWindows()
img_for_optional = np.hstack((img_hw,img_hw2))
cv2.imshow('optional',img_for_optional)
cv2.waitKey(0)
cv2.destroyAllWindows()
print('可以發現 transformation matrix 跟雙線性差值的結果會有一點點差異，但也只有一點點')
print('--------------------------------------------------------------------------------------------------------------')
# -------------------------------------------------------------------------------------------------------------------- #
# 是不是可以直接畫好框框然後再縮放?
picture_for_final = cv2.imread("lena.jpg",cv2.IMREAD_COLOR)
picture_for_final = cv2.cvtColor(picture_for_final,cv2.COLOR_BGR2HSV)
picture_for_final[...,2] = cv2.equalizeHist(picture_for_final[...,2])
picture_for_final_change_value = cv2.cvtColor(picture_for_final, cv2.COLOR_HSV2BGR)
picture_for_final_projector = picture_for_final_change_value[:,::-1,:]
# 看起來在畫上框架後，圖片的type會改變
# 所以還要再重新定義一次type
picture_for_final_projector = picture_for_final_projector.astype('uint8')
picture_for_final_retangle = cv2.rectangle(picture_for_final_projector,(60,40),(420,510),(0,0,255),3)
picture_for_final_scale =  cv2.resize(picture_for_final_retangle, None, fx=0.5, fy=0.5,interpolation=cv2.INTER_AREA)
picture_for_final_123 = cv2.cvtColor(picture_for_final_scale, cv2.COLOR_BGR2RGB)
cv2.imshow('FINAL1',picture_for_final_projector)
cv2.waitKey(0)
cv2.destroyAllWindows()
# 他們尺寸不同不能一起
# Final = np.hstack((img_hw,picture_for_final_projector))
# cv2.imshow('FINAL2',Final)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
img_hw = cv2.cvtColor(img_hw, cv2.COLOR_BGR2RGB)
picture_for_final_projector = cv2.cvtColor(picture_for_final_projector, cv2.COLOR_BGR2RGB)
plt.figure()
plt.subplot(2,1,1)
plt.imshow(img_hw)
plt.axis('off')
plt.subplot(2,1,2)
plt.imshow(picture_for_final_projector)
plt.axis('off')
plt.show()
print('可以發現這樣子有點跑掉')
# -------------------------------------------------------------------------------------------------------------------- #


















































































