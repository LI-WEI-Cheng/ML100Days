from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import random
import imgaug as ia
from imgaug import augmenters as iaa
from PIL import Image
import os
import pickle
import glob
import pandas as pd

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import utils
###
# ImageDataGenerator 可以做到許多的圖像增強:
# featurewise_center：輸入為Boolean(True or False)，以每一張feature map為單位將平均值設為0
# featurewise_std_normalization: 輸入為Boolean(True or False) ，以每一張feature map為單位將數值除以其標準差(上述兩步驟就是我們常見的Standardization)
# Standardization 的用途在於使不同的輸入影像有相似的資料分佈範圍，這樣在收斂時會比較快，也較容易找到 Global Minimum。
# ImageDataGenerator中常見的Augmentation(輸入形式、內容)：
# zca_whitening: Boolean，透過ZCA取出重要特徵(詳見：ZCA介紹)
# rotation_range：整數值，控制隨機旋轉角度
# width_shift_range：「浮點、整數、一維數」，圖像寬度上隨機偏移值
# height_shift_range：「浮點、整數、一維數」，圖像高度上隨機偏移值
# shear_range：浮點數，裁切範圍
# zoom_range：浮點數或範圍，隨機縮放比例
# horizontal_flip: Boolean，隨機水平翻轉
# vertical_flip:Boolean，隨機垂直翻轉
# rescale: 數值，縮放比例
# dtype：輸出資料型態
###
# 定義使用的Augmentation
img_gen = ImageDataGenerator( featurewise_center=True,featurewise_std_normalization=True,rotation_range=30,width_shift_range=0.1,
                                            height_shift_range=0.1,shear_range=0.1,zoom_range=0.1,horizontal_flip=True,vertical_flip=False,dtype=np.float32)

width=224
height=224
batch_size=4

img = cv2.imread('E:\ML100DAYS\ML100Days\Tano.jpg')
print(img.shape)
img = cv2.resize(img, (224,224))##改變圖片尺寸
print(img.shape)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #Cv2讀進來是BGR，轉成RGB
img_origin=img.copy()
img= np.array(img, dtype=np.float32)
img_combine=np.array([img,img,img,img],dtype=np.uint8) ##輸入generator要是四維，(224,224,3)變成(4,224,224,3)
batch_gen = img_gen.flow(img_combine, batch_size=4)

# assert 用於判斷一個表達式，若無滿足該表達式的條件，則直接觸發異常狀態，而不會接續執行後續的程式碼。
# next（）返回迭代器的下一個項目。
# next（）函數要和生成迭代器的iter（）函數一起使用。
assert next(batch_gen).shape==(batch_size, width, height, 3)

plt.figure(figsize=(15,10))

i = 1
for batch in batch_gen:
    plt.subplot(1, 5, 1)
    plt.imshow(img_origin)
    plt.subplot(1, 5, i+1)
    plt.imshow(batch[0, :, :, :].astype(np.uint8))
    plt.imshow(batch[1, :, :, :].astype(np.uint8))
    plt.imshow(batch[2, :, :, :].astype(np.uint8))
    plt.imshow(batch[3, :, :, :].astype(np.uint8))
    plt.axis('off')
    i += 1
    if i > 4:
        break  # or the generator would loop infinitely
plt.show()

# -------------------------------------------------------------------------------------------------------------------- #
# ##
# # 導入ImageDataGenerator到Keras訓練中
# ##
#
# # Training Generator
# train_datagen = ImageDataGenerator(rescale=2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
# # Test Generator，只需要Rescale，不需要其他增強
# test_datagen = ImageDataGenerator(rescale=1. / 255)
#
# # 將路徑給Generator，自動產生Label
# training_set = train_datagen.flow_from_directory(x_train,
#                                                  target_size=(64, 64),
#                                                  batch_size=32,
#                                                  class_mode='categorical')
#
# test_set = test_datagen.flow_from_directory(x_test,
#                                             target_size=(64, 64),
#                                             batch_size=32,
#                                             class_mode='categorical')
#
# model = tf.saved_model.load('E:\ML100DAYS\ML100Days\my_training_modle')  # 指定讀取路徑，讀取模型檔
# # 訓練
# model.fit_generator(training_set, steps_per_epoch=250, nb_epoch=25,
#                          validation_data=valid_set, validation_steps=63)
#
# from keras.preprocessing import image as image_utils
# test_image = image_utils.load_img('E:\ML100DAYS\ML100Days\dog.jpg', target_size=(224, 224))
# test_image = image_utils.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis=0)
#
# result = model.predict_on_batch(test_image)
# #預測新照片
# from keras.preprocessing import image as image_utils
# test_image = image_utils.load_img('dataset/new_images/new_picture.jpg', target_size=(224, 224))
# test_image = image_utils.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis=0)
#
# result = classifier.predict_on_batch(test_image)

# -------------------------------------------------------------------------------------------------------------------- #
##
# 如何使用Imgaug
##

img123 = cv2.imread('E:\ML100DAYS\ML100Days\Tano.jpg')
img123 = cv2.resize(img123, (224,224))##改變圖片尺寸
img123 = cv2.cvtColor(img123, cv2.COLOR_BGR2RGB) #Cv2讀進來是BGR，轉成RGB
img_origin123=img123.copy()
img123= np.array(img123, dtype=np.float32)

images123 = np.random.randint(0, 255, (5, 224, 224, 3), dtype=np.uint8)##創造一個array size==(5, 224, 224, 3)

flipper = iaa.Fliplr(1.0) #水平翻轉機率==1.0
images123[0] = flipper.augment_image(img123)

vflipper = iaa.Flipud(0.4) #垂直翻轉機率40%
images123[1] = vflipper.augment_image(img123)

blurer = iaa.GaussianBlur(3.0)
images123[2] = blurer.augment_image(img123) # 高斯模糊圖像( sigma of 3.0)

translater = iaa.Affine(translate_px={"x": -16}) #向左橫移16個像素
images123[3] = translater.augment_image(img123)

scaler = iaa.Affine(scale={"y": (0.8, 1.2)}) # 縮放照片，區間(0.8-1.2倍)
images123[4] = scaler.augment_image(img123)

i=1
plt.figure(figsize=(10,20))
for img123 in images123:
    plt.subplot(1, 6, 1)
    plt.imshow(img_origin.astype(np.uint8))
    plt.subplot(1, 6, i+1)
    plt.imshow(img123.astype(np.uint8))
    plt.axis('off')
    i+=1
plt.show()

# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

# 包裝自定義Augmentation 與 Imgaug Augmentation
'''隨機改變亮度'''


class RandomBrightness(object):
    '''Function to randomly make image brighter or darker
    Parameters
    ----------
    delta: float
        the bound of random.uniform distribution
    '''

    def __init__(self, delta=16):
        assert 0 <= delta <= 255
        self.delta = delta

    def __call__(self, image):
        delta = random.uniform(-self.delta, self.delta)
        if random.randint(0, 1):
            image = image + delta
        # 也就是說，截取的意思，超出的部分就把它強置為邊界部分。
        # numpy. clip ( a ,  a_min ,  a_max ,  out=None )
        image = np.clip(image, 0.0, 255.0)
        return image


'''隨機改變對比'''


class RandomContrast(object):
    '''Function to strengthen or weaken the contrast in each image
    Parameters
    ----------
    lower: float
        lower bound of random.uniform distribution
    upper: float
        upper bound of random.uniform distribution
    '''

    def __init__(self, lower=0.5, upper=1.5):
        assert upper >= lower, "contrast upper must be >= lower."
        assert lower >= 0, "contrast lower must be non-negative."
        self.lower = lower
        self.upper = upper

    def __call__(self, image):
        alpha = random.uniform(self.lower, self.upper)
        if random.randint(0, 1):
            image = image * alpha
        image = np.clip(image, 0.0, 255.0)
        return image


'''包裝所有Augmentation'''


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image


'''包裝Imgaug'''


class ImgAugSequence(object):
    def __init__(self, sequence):
        self.sequence = sequence

    def __call__(self, image):
        image = self.sequence.augment_image(image)

        return image


class TrainAugmentations(object):
    def __init__(self):
        # Define imgaug.augmenters Sequential transforms
        sometimes = lambda aug: iaa.Sometimes(0.4, aug)  # applies the given augmenter in 50% of all cases

        img_seq = iaa.Sequential([
            sometimes(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.03 * 255), per_channel=0.5)),
            sometimes(iaa.ContrastNormalization((0.5, 2.0), per_channel=1), ),
            sometimes(iaa.Sharpen(alpha=(0, 0.2), lightness=(0.1, 0.4))),  # sharpen images
            sometimes(iaa.Emboss(alpha=(0, 0.3), strength=(0, 0.5))),  # emboss images
        ], random_order=True)

        self.aug_pipeline = Compose([
            RandomBrightness(16),  # make image brighter or darker
            RandomContrast(0.9, 1.1),  # strengthen or weaken the contrast in each image
            ImgAugSequence(img_seq),
        ])

    def __call__(self, image):
        image = self.aug_pipeline(image)
        return image


Augmenation = TrainAugmentations()

##輸入照片
img = cv2.imread('E:\ML100DAYS\ML100Days\Tano.jpg')
img = cv2.resize(img, (224,224))##改變圖片尺寸
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #Cv2讀進來是BGR，轉成RGB

output=Augmenation(img)

##畫出來
plt.figure(figsize=(10,10))
plt.imshow(output.astype(np.uint8))
plt.axis('off')
plt.show()



