from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import random
import string
from keras import backend as K
from keras.models import *
from keras.layers import *
import keras
#
# 補充
# generate_image(chars) 生成驗證碼的核心方法，生成chars內容的驗證碼圖片的Image對象。
# create_captcha_image(chars, color, background) generate_image的實現方法，可以通過重寫此方法來實現自定義驗證碼樣式。
# create_noise_dots(image, color, width=3, number=30) 生成驗證碼干擾點。
# create_noise_curve(image, color) 生成驗證碼干擾曲線。
#

# captcha 是驗證碼的意思!
# 驗證碼包含0-10數字以及26個英文字母
characters = string.digits + string.ascii_uppercase
print(characters)
# 設定產生圖片尺寸，以及總類別，n_class之所以要加一是為了留一個位置給Blank
width, height, n_len, n_class = 170, 80, 4, len(characters)+1
#
# 設定產生驗證碼的generator
# ImageCaptcha(width=160, height=60, fonts=None, font_sizes=None) 類實例化時，還可傳入四個參數:
# width: 生成驗證碼圖片的寬度，默認為160個像素；
# height： 生成驗證碼圖片的高度，默認為60個像素；
# fonts：字體文件路徑，用於生成驗證碼時的字體，默認使用模塊自帶DroidSansMono.ttf字體，
# 你可以將字體文件放入list或者tuple傳入,生成驗證碼時將隨機使用;
# font_sizes：控制驗證碼字體大小，同fonts一樣，接收一個list或者tuple,隨機使用。
#
generator = ImageCaptcha(width=width, height=height, fonts=None, font_sizes=None)
print(generator)
# 我們先練習固定長度4個字的驗證碼
random_str = ''.join([random.choice(characters) for j in range(4)])
img = generator.generate_image(random_str)
plt.imshow(img)
plt.title(random_str)
plt.show()
# ---------------------------------------------------------------------------------------

rnn_size = 128
input_tensor = Input((height,width, 3))
x = input_tensor

for i in range(4):
    x = Convolution2D(32, 3, 3, activation='relu')(x)
    x = keras.layers.BatchNormalization(axis=-1)(x)
    x = Convolution2D(32, 3, 3, activation='relu')(x)
    x = keras.layers.BatchNormalization(axis=-1)(x)
    if i < 3:
        x = MaxPooling2D(pool_size=(2, 2))(x)
    else:
        x = MaxPooling2D(pool_size=(2, 1))(x)

# 記錄輸出CNN尺寸，loss部分需要這個資訊
# conv_shape=(Batch_size,輸出高度,輸出寬度,輸出深度)
conv_shape = x.get_shape()
print(conv_shape[0])
print(conv_shape[1])
print(conv_shape[2])
print(conv_shape[3])
# 從(Batch_size,輸出高度,輸出寬度,輸出深度)變成(Batch_size,輸出寬度,輸出深度*輸出高度)
x = Reshape(target_shape=(int(conv_shape[2]), int(conv_shape[1]*conv_shape[3])))(x)
conv_shape2 = x.get_shape()
print(conv_shape2)

x = Dense(32, activation='relu')(x)
x = Dropout(0.25)(x)
x = Dense(n_class, activation='softmax')(x)


# 包裝用來預測的model
base_model = Model(input=input_tensor, output=x)

# -------------------------------------------------------------------------------------------------------------- #

# CTC Loss需要四個資訊，分別是
# Label
# 預測
# CNN OUTPUT寬度
# 預測影像所包含文字長度

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
# 設定要給CTC Loss的資訊
# n_len : 總類別
labels = Input(name='the_labels', shape=[n_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([x, labels, input_length, label_length])
# 這裡的model是用來計算loss
model = Model(input=[input_tensor, labels, input_length, label_length], output=[loss_out])
# 之所以要lambda y_true, y_pred: y_pred是因為我們的loss已經包在網路裡，會output:y_true, y_pred，而我們只需要y_pred
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='SGD')
model.summary()

# 設計generator產生training data
# 產生包含要給loss的資訊
# X=輸入影像
# np.ones(batch_size)*int(conv_shape[2])=CNN輸出feature Map寬度
# np.ones(batch_size)*n_len=字串長度(可浮動)

def gen(batch_size=128):
    X = np.zeros((batch_size,height, width, 3), dtype=np.uint8)
    y = np.zeros((batch_size, n_len), dtype=np.uint8)
    while True:
        generator = ImageCaptcha(width=width, height=height)
        for i in range(batch_size):
            random_str = ''.join([random.choice(characters) for j in range(4)])
            X[i] = np.array(generator.generate_image(random_str))
            y[i] = [characters.find(x) for x in random_str]
        yield [X, y,np.ones(batch_size)*int(conv_shape[2]), np.ones(batch_size)*n_len], np.ones(batch_size)
next_ge=gen(batch_size=1)
test_ge=next(next_ge)
plt.imshow(test_ge[0][0][0])
print('Label: ',test_ge[0][1])
print('CNN輸出寬度: ',test_ge[0][2])
print('字串長度(可浮動): ',test_ge[0][3])
model.fit_generator(gen(32), steps_per_epoch=300, epochs=60)





