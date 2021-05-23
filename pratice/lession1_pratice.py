import cv2 as cv
img1 = cv.imread('unnamed.jpg',cv.IMREAD_COLOR) # 透過 imread 載入圖片，載入包含 Blue, Green, Red 三個 channel 的彩色圖片，
img2 = cv.imread('unnamed.jpg',cv.IMREAD_GRAYSCALE)#載入灰階格式的圖片
img3 = cv.imread('unnamed.jpg',cv.IMREAD_UNCHANGED)#載入圖片中所有 channel
a = type(img1)#你可以透過 type，發現 OpenCV 是用 numoy 表示圖片格式
print(a)
# #透過 OpenCV 顯示圖片cv.imshow('rgb1'-->要顯示圖片的視窗名子, img1-->要顯示的圖片)
cv.imshow('rgb1', img1)
cv.imshow('rgb2', img2)
cv.imshow('rgb3', img3)
cv.waitKey(0) # cv2.waitKey([delay]) --> delay – Delay in milliseconds. 0 is the special value that means “forever”.
# 所以給10，會變成說10豪秒後他會關閉
cv.destroyAllWindows()
# Destroys all of the HighGUI windows.
# HightGui是一个可以移植的图形工具包。OpenCV将与操作系统，文件系统，摄像机之类的硬件进行交互的一些函数纳入HighGui库中
######################################################################################################################
#作業範例
img_path = 'unnamed.jpg'

# 以彩色圖片的方式載入
img11 = cv.imread(img_path, cv.IMREAD_COLOR)

# 以灰階圖片的方式載入
img_gray = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

# 為了要不斷顯示圖片，所以使用一個迴圈
while True:
    # 顯示彩圖
    cv.imshow('bgr', img11)
    # 顯示灰圖
    cv.imshow('gray', img_gray)

    # 直到按下 ESC 鍵才會自動關閉視窗結束程式
    k = cv.waitKey(0)
    if k == 27:
        cv.destroyAllWindows()
        break
# 在这个程序中,我们告诉OpenCv等待用户触发事件,等待时间为無限ms，如果在这个时间段内, 用户按下ESC(ASCII码为27),则跳出循环
# 如果设置waitKey(0),则表示程序会无限制的等待用户的按键事件




