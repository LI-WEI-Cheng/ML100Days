# 改變 color space 來調整、操作飽和度 :
# 練習時，以HSV做練習，這次換成HSL，所以在取channel方面不同，但同樣條飽和度則 : img[:,:,-1]
# 若要調整其他亮度或色相，則取其他
# ---------------------------------------------------　#
import cv2
import numpy as np
picture = cv2.imread('lena.jpg',cv2.IMREAD_COLOR)
img1 = cv2.cvtColor(picture, cv2.COLOR_BGR2HLS)
img1 = img1.astype('float32')
img1[...,-1] = img1[...,-1]/255-0.2
img1[img1[...,-1] < 0] = 0
img1[...,-1] = img1[...,-1]*255
img1 = img1.astype('uint8')
img_down = cv2.cvtColor(img1, cv2.COLOR_HLS2BGR)

# ---------------------------------------------------　#
img0 = cv2.cvtColor(picture, cv2.COLOR_BGR2HLS)
img0 = img0.astype('float32')
img0[...,-1] = img0[...,-1]/255
img0[img0[...,-1] < 0] = 0
img0[...,-1] = img0[...,-1]*255
img0 = img0.astype('uint8')
img_same = cv2.cvtColor(img0, cv2.COLOR_HLS2BGR)

# ---------------------------------------------------　#
img2 = cv2.cvtColor(picture, cv2.COLOR_BGR2HLS)
img2 = img2.astype('float32')
img2[...,-1] = img2[...,-1]/255+0.2
img2[img2[...,-1] > 1] = 1
img2[...,-1] = img2[...,-1]*255
img2 = img2.astype('uint8')
img_up = cv2.cvtColor(img2, cv2.COLOR_HLS2BGR)
img_combine = np.hstack((img_down,img_same,img_up))


# ---------------------------------------------------　 ---------------------------------------------------　 ---------------------------------------------------　#
# 實作直方圖均衡(2種):
# ---------------------------------------------------　#
picture1 = cv2.imread('lena.jpg',cv2.IMREAD_COLOR)
img_HSV_HE = cv2.cvtColor(picture1, cv2.COLOR_BGR2HSV)
img_HSV_HE_equal = cv2.equalizeHist(img_HSV_HE[...,2])
img_HSV_HE[...,2] =  img_HSV_HE_equal
img_final = cv2.cvtColor(img_HSV_HE , cv2.COLOR_HSV2BGR)

# ---------------------------------------------------　#
img_rgb = cv2.imread('lena.jpg',cv2.IMREAD_COLOR)
img_rgb[...,0] = cv2.equalizeHist(img_rgb[...,0])
img_rgb[...,1] = cv2.equalizeHist(img_rgb[...,1])
img_rgb[...,2] = cv2.equalizeHist(img_rgb[...,2])
img_all_change_combine = np.hstack((picture1,img_final,img_rgb))


# ---------------------------------------------------　 ---------------------------------------------------　 ---------------------------------------------------　#
#alpha/ beta 調整對比 / 明亮
# ---------------------------------------------------　#
# 下列透過foe迴圈來對逐一每個pixel調整
alpha=2.0
beta=0
for b in range (picture1.shape[0]):
    for g in range(picture1.shape[1]):
        for r in range  (picture1.shape[2]):
            # 截取的意思，超出的部分就把它強置為邊界部分: 限制範圍在0-255
            picture1[b,g,r] = np.clip(alpha*picture1[b,g,r]+beta,0,255)
pic1 = picture1
# ---------------------------------------------------　#
picture2 = cv2.imread('lena.jpg',cv2.IMREAD_COLOR)
alpha2=1.0
beta2=50
for b in range (picture2.shape[0]):
    for g in range(picture2.shape[1]):
        for r in range  (picture2.shape[2]):
            # 截取的意思，超出的部分就把它強置為邊界部分: 限制範圍在0-255
            picture2[b,g,r] = np.clip(alpha2*picture2[b,g,r]+beta2,0,255)
pic2 = picture2
img_all_change_apha_beta_combine = np.hstack((picture,pic1,pic2))

# ---------------------------------------------------　 ---------------------------------------------------　 ---------------------------------------------------　#
import matplotlib.pyplot as plt # plt 用于显示图片 # 等於from matplotlib import pyplot as plt
# BGR轉RGB
img_111 = cv2.cvtColor(img_combine, cv2.COLOR_BGR2RGB)
img_222 = cv2.cvtColor(img_all_change_combine, cv2.COLOR_BGR2RGB)
img_333 = cv2.cvtColor(img_all_change_apha_beta_combine, cv2.COLOR_BGR2RGB)
plt.figure()
plt.subplot(3,1,1)
plt.imshow(img_111)
plt.axis('off')
plt.subplot(3,1,2)
plt.imshow(img_222)
plt.axis('off')
plt.subplot(3,1,3)
plt.imshow(img_333)
plt.axis('off')
plt.show()

# cv2.imshow('ALL', img_combine)
# cv2.waitKey(0)
# cv2.imshow('CHANGE final-2',img_all_change_combine)
# cv2.waitKey(0)
# cv2.imshow('Adjust',img_all_change_apha_beta_combine)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

















