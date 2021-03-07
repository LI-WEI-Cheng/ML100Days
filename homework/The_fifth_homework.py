import cv2
import numpy as np
import matplotlib.pyplot as plt
# -------------------------------------------------------------------------------------------------#
# Hint: 人物原始邊框座標 (60, 40), (420, 510)
# 請根據 Lena 圖做以下處理
# 對明亮度做直方圖均衡處理
# 水平鏡像 + 縮放處理 (0.5 倍)
# 畫出人物矩形邊框
# -------------------------------------------------------------------------------------------------#
picture_for_example = cv2.imread("lena.jpg",cv2.IMREAD_COLOR)
picture_for_example = cv2.cvtColor(picture_for_example,cv2.COLOR_BGR2HSV)
picture_for_example[...,2]=  cv2.equalizeHist(picture_for_example[...,2])
picture_for_example_change_value = cv2.cvtColor(picture_for_example, cv2.COLOR_HSV2BGR)
picture_for_example_projector = picture_for_example_change_value[:,::-1,:]
# 如果是要縮小圖片的話，通常 INTER_AREA 使用效果較佳。
picture_for_example_scale =  cv2.resize(picture_for_example_projector, None, fx=0.5, fy=0.5,interpolation=cv2.INTER_AREA)
picture_for_example_retangle = cv2.rectangle(picture_for_example_scale,(30,25),(230,250),(0,0,255),3)
cv2.imshow('Last Exmple',picture_for_example_retangle)
cv2.waitKey(0)
cv2.destroyAllWindows()
# -------------------------------------------------------------------------------------------------#

