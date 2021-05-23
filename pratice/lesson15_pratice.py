# CNN分類器
# ------------------------------------------------------------------------------------------------------------------- #
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from sklearn.externals import joblib
import tensorflow as tf


# ------------------------------------------------------------------------------------------------------------------- #
# 匯入 Keras 的 cifar10 模組
from tensorflow.keras.datasets import cifar10
import numpy as np
from tensorflow.keras import utils
np.random.seed(10)
# 載入 Cifar-10 資料集
(x_train_image, y_train_label), (x_test_image, y_test_label) = cifar10.load_data()
# ------------------------------------------------------------------------------------------------------------------- #
# 訓練集之數字圖片陣列共有 50000 筆 32*32 解析度的彩色 RGB 圖片
# 測試集則有 10000 筆 32*32 解析度的彩色 RGB 圖片
print('train image numbers=', len(x_train_image))  #顯示訓練圖片筆數 : 5 萬筆
print('train label numbers=', len(y_train_label))  #顯示訓練標籤筆數 : 5 萬筆
print('test image numbers=', len(x_test_image))    #顯示測試標籤筆數 : 1 萬筆
print('test label numbers=', len(y_test_label))    #顯示測試標籤筆數 : 1 萬筆
# -------------------------------------- #
# 不論是訓練集還是測試集, 傳回值 x_train_image 與 x_test_image 均為 3 維陣列,
# 利用索引 0~49999 可以查詢 5 萬筆訓練集圖片之內容, 測試集圖片的索引範圍則為 0~9999.
# -------------------------------------- #
# 訓練集的第一張圖片資料為陣列，每一個畫素以 1*3 向量 [R G B] 形式來表示,
# 因為圖片是32*32*3，所以光最左上角的一小格就會跑出[a b c]三維向量，代表RGB
# 所以再第一行會有32個向量，共32行
# 一張圖總共有 32*32=1024 個向量, 有 32*32*3=3072 個數字
print('x_train_image[0]=', x_train_image[0])
# -------------------------------------- #
# 訓練集第一張圖的第一列畫素為 x_train_image[0][0], 由 32 個一維向量組成 :
print('x_train_image[0][0]=', x_train_image[0][0])
# -------------------------------------- #
# 訓練集第一張圖片的第一個畫素 x_train_image[0][0][0] 如下 :
print('x_train_image[0][0][0]=', x_train_image[0][0][0])
# -------------------------------------- #
# 顯示全部訓練集標籤
print('y_train_label=', y_train_label)
# -------------------------------------- #
# 第一張圖片是分類 6 (青蛙)
print('y_train_label[0][0]=', y_train_label[0][0])
# ------------------------------------------------------------------------------------------------------------------- #
# 前處理的方式，同樣分成 Training data 以及 Test data，然後圖片的資訊以及 Label 也分開，分別傳入四個變數。
x_train = x_train_image.astype('float32')/255
x_test = x_test_image.astype('float32')/255
y_train = utils.to_categorical(y_train_label)
y_test = utils.to_categorical(y_test_label)

# 透過卷積核 (Kernels) 滑動對圖像做訊息提取，並藉由步長 (Strides) 與填充 (Padding) 控制圖像的長寬。
# 卷積核又稱為 Filter、Kernel、feature detector ，

# 起始值：圖中就是一個 3*3的 Kernel，其中的值就是我們要訓練的權重，通常用常態分佈隨機產生，再經由訓練更新。
# 張數：控制張數主要就是控制學習的參數量，常見是16、32 或 64，如果使用16張 Kernels 就能達到很好的效果，也就不需要浪費額外的參數去增加學習與計算量。
# 張數其實也就是下一層的輸入數量啦!

# Kernel 用來學習圖像的特徵，Kernel 的張數 (filters) 決定學習的參數量，Kernel 大小決定 Kernels 的特徵接受域，也就是看到特徵的大小。
# 『最大池化( Max Pooling )』 取出 Kernel 內最大的值
# 『平均池化( Average Pooling )』取出 Kernel 內的平均值.
# 全連接層( FC ) : 「其目的主要是要是為了利用全連接層的神經元做為分類器各類別的代表機率。」
# pool_size :整數，最大池化的窗口大小。
# strides :整數，或者是None。作為縮小比例的因數。例如，2會使得輸入張量縮小一半。如果是None，那麼默認值是pool_size。

# 攤平( Flatten ) : 攤平其實就是將大於 1 維的 Tensor 拉平為二維( batch size, channels )，通常經過 CNN 的 Feature Map 為四維，拉平為二維才能夠與全連接層合併。

model = Sequential()
model.add(Convolution2D(filters=64, kernel_size=3, input_shape=(32, 32, 3), activation='relu', padding='same'))
model.add(Convolution2D(filters=64, kernel_size=3, input_shape=(32, 32, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=2))

model.add(Convolution2D(filters=128, kernel_size=3, activation='relu', padding='same'))
model.add(Convolution2D(filters=128, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=2))

model.add(Convolution2D(filters=128, kernel_size=3, activation='relu', padding='same'))
model.add(Convolution2D(filters=128, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
# Dropout就是每次訓練按概率拿走一部分神經元，只在訓練時使用。
# Dropout觀念相當直觀，就是在每次 Forward Propagation 時隨機關閉特定比例的神經元，避免模型 Overfitting。
model.add(Dropout(rate=0.25))
model.add(Dense(10, activation='softmax'))
print(model.summary())
# 以compile函數定義損失函數(loss)、優化函數(optimizer)及成效衡量指標(mertrics)。
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# x：輸入數據。y：標籤。batch_size：整數。verbose：日誌顯示，0為不在標準輸出流輸出日誌信息，1為輸出進度條記錄，2為每個epoch輸出一行記錄
model.fit(x_train, y_train, epochs=1, batch_size=64, verbose=1)
# , verbose=1 輸出進度條的意思
# 顯示訓練成果(分數)
loss, accuracy = model.evaluate(x_test, y_test)
print('Test:')
print('Loss:', loss)
print('Accuracy:', accuracy)
#save model
export_path = 'E:\ML100DAYS\ML100Days\my_training_modle'  # 指定路徑
tf.saved_model.save(model, export_path)  # 存檔











