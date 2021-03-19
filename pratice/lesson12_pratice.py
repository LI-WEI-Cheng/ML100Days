# -------------------------------------------------------------------------------------------------------------------- #
# 步長(Strides)：
# 圖中 Kernel 的步長 (Strides) 在 height、 width 均為 1，可以看到黃色 Kernel 往右、下都是一格像素。
# 控制 Kernel 在圖像不同 Dimention 上移動的距離。
# 運用不同的步長(1,1)與(2,2)，可以發現輸出的 Feature map 尺寸也有所不同。

# Keras Convolution2D：
# 其中有一個可調參數為 Strides，可以針對Height 、 Width 賦予不同的值，藉此控制輸出 Feature map 高、寬的尺度變化。
# 在操作 Keras 的 Convolution2D 時，會發現默認為『 Valid 』而一多數人會用『 SAME 』，這是什麼意思呢？
#   Valid：
#          就是不去 Padding，多的像素直接捨去。
#          像是上圖中，可以想像原圖為13*13，kernel 大小為6*6，步長為(5,5)，
#          當Kernel要跨出第二步時，只剩下 2 個像素(12、13)，就直接捨去。
#   SAME：
#          透過補邊讓輸出長寬==原圖長寬/Strides，什麼意思呢？
#          假如我們使用Strides=(1,1)，那麼不管使用多大的Kernel，輸出 Feature map 的寬、高等於輸入影像的寬高。

# 填充 (Padding) ：
# 沒有使用任何 Padding，因此可以看到原圖周圍並沒有補 0 的像素，而輸出的 Feature map 長寬也下降。
# Padding 的用途主要在於避免圖像尺寸下降，而為了避免干擾圖像資訊，
# 通常Padding為補 0 的像素，而 Padding=1 就是在圖像周圍補一圈值為 0 的像素，也就是圖中灰色的區域。
#
# -------------------------------------------------------------------------------------------------------------------- #
# 作業內容：
# 運用Keras搭建簡單的Convolution2D Layer，調整Strides與Padding參數計算輸出feature map大小。
# -------------------------------------------------------------------------------------------------------------------- #
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Input, Dense
from keras.layers import Activation, Dense
classifier=Sequential()
# --------------------------------------------------- #
# Convolution2D(捲積核數量,幾乘幾大小,strides=1, padding='valid', dilation_rate=1, activation=None, use_bias=True,
# kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
# activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
# dilation_rate=1 : 整數或由單一整數組成的列表/元組，指定dilated卷積中的膨脹比例。
# activation=None : 激活函數
# use_bias=True : 布爾值，是否使用替換項
# kernel_initializer : 權值初始化方法，為預定義初始化方法名的字符串，或用於初始化權重的初始化器。
# bias_initializer='zeros' : 權值初始化方法，為預定義初始化方法名的字符串，或用於初始化權重的初始化器。
# kernel_regularizer：施加在權重上的正則項
# bias_regularizer：施加在校正向量上的正則項
# activity_regularizer：施加在輸出上的正則項
# kernel_constraints：施加在權重上的約束項
# bias_constraints：施加在收縮上的約束項
# --------------------------------------------------- #
# 建立模型
classifier=Sequential()
# --------------------------------------------------- #
# 卷積層：
# Kernel size 3*3，用32張，輸入大小28*28*1
# data_format：channels_first就會是(None, 1, 28, 28)，如果channels_last就會是(None, 28, 28, 1)，這在GPU運算上速度會有差別
# 卷積完，通常會做激勵函數relu，正數保持原數、負數會變成0，
# 圖片的神經網路通常正數才是我們要的參數，幫助神經網路提高效率、減少不必要的計算
classifier.add(Convolution2D(
    batch_input_shape=(None, 1, 28, 28),
    filters=32,
    kernel_size=5,
    strides=1,
    padding='same',     # Padding method
    data_format='channels_first',
))
classifier.add(Activation('relu'))
# --------------------------------------------------- #
# 做一個池化層
# MaxPooling2D：最大池化層
# pool_size=2：想像每2x2的像素，四個像素中哪個數字最大，就保留這格的值
# strides=2：每次跨兩步，輸入張量會變成原來的一半。想像整張照片會縮小成原來寬的1/2、長的1/2
classifier.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding='same',
    data_format='channels_first',
))
# model = Model(inputs=inputs, outputs=x)
classifier.summary()
# -------------------------------------------------------------------------------------------------------------------- #
classifier.add(Convolution2D(
    batch_input_shape=(None, 1, 28, 28),
    filters=32,
    kernel_size=5,
    strides=2,
    padding='same',     # Padding method
    data_format='channels_first',
))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding='valid',
    data_format='channels_first',
))
classifier.summary()



























