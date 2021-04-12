# InceptionV1-Inception Layer : 採用1*1的捲積做降維
# 由於信息位置的巨大差異，為卷積操作選擇合適的卷積核大小就比較困難
# 信息分佈更全局性的圖像偏好較大的卷積核
# 信息分佈比較局部的圖像偏好較小的卷積核
# --> Inception概念：結合不同大小的 Kernels借以獲得不同尺度的訊息。
# 圖中展示了經典的 Inception架構，接在 Feature Maps 後一共有四條分支，其中三條先經過 1*1 kernel 的壓縮，
# 這樣做的意義主要是為了控制輸出 Channels 的深度，並同時能增加模型的非線性，一條則是先通過3*3 kernel 的 Maxpooling。
# 為了確保輸出 Feature Map 在長寬上擁有一樣尺寸，我們就要借用 Padding 技巧，
# 1*1 kernel 輸出大小與輸入相同，而 3*3、5*5 kernel 則分別設定補邊值為 1、2，
# 在 tensorflow、Keras 中最快的方式就是設定padding=same，就能在步伐為1時確保輸出尺寸維持相同。

# 1*1 kernel 的壓縮其實就是一般的卷積，然而它的好處在於能用相當少的參數量，達到壓縮特徵圖深度的目的，
# 舉個例子來說，當輸入 Feature Map 為 (batch_size,14,14,192)，要將其壓縮為(batch_size,14,14,n)，
# 我們只需要 1*1*192*n+n 個參數量，當然同樣事情也可以用 3*3 kernel 達成，但參數量就會變為 3*3*192*n+n。
import numpy as np
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras import backend as K
from keras import layers
from keras.layers import BatchNormalization
from keras.layers import Activation

def Conv2d_bn(x,filters,kernel_size,padding='same',strides=(1, 1),normalizer=True,activation='relu',name=None):
    if name is not None:
        conv_name = name + '_conv'
        bn_name = name + '_bn'
        act_name = name + '_act'
    else:
        conv_name = None
        bn_name = None
        act_name = None

    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Conv2D(
            filters, kernel_size,
            strides=strides, padding=padding,
            use_bias=False, name=conv_name)(x)

    if normalizer:
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    if activation:
        x = Activation(activation, name=act_name)(x)
    return x

def InceptionV1_block(x, specs,channel_axis, name):
    (br0, br1, br2, br3) = specs   # ((64,), (96,128), (16,32), (32,))
    branch_0 = Conv2d_bn(x, br0[0], (1, 1), name=name+"_Branch_0")

    branch_1 = Conv2d_bn(x, br1[0], (1, 1), name=name+"_Branch_1")
    branch_1 = Conv2d_bn(branch_1, br1[1], (3, 3), name=name+"_Branch_1_1")

    branch_2 = Conv2d_bn(x, br2[0], (1, 1), name=name+"_Branch_2")
    branch_2 = Conv2d_bn(branch_2, br2[1], (5, 5), name=name+"_Branch_2_1")

    branch_3 = MaxPooling2D( (3, 3), strides=(1, 1), padding='same', name=name+"_Branch_3")(x)
    branch_3 = Conv2d_bn(branch_3, br3[0], (1, 1), name=name+"_Branch_3_1")

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name=name+"_Concatenated")
    return x
img_input = Input(shape=(224,224,1))
x=InceptionV1_block(img_input, ((64,), (96,128), (16,32), (32,)), 3, 'Block_1')
print(x)

def InceptionV3_block(x, specs,channel_axis, name):
    (br0, br1, br2, br3) = specs   # ((64,), (96,128), (16,32), (32,))
    branch_0 = Conv2d_bn(x, br0[0], (1, 1), name=name+"_Branch_0")

    branch_1 = Conv2d_bn(x, br1[0], (1, 1), name=name+"_Branch_1")
    branch_1 = Conv2d_bn(branch_1, br1[1], (1, 3), name=name+"_Branch_1_1")
    branch_1 = Conv2d_bn(branch_1, br1[1], (3, 1), name=name+"_Branch_1_2")

    branch_2 = Conv2d_bn(x, br2[0], (1, 1), name=name+"_Branch_2")
    branch_2 = Conv2d_bn(branch_2, br2[1], (1, 5), name=name+"_Branch_2_1")
    branch_2 = Conv2d_bn(branch_2, br2[1], (5, 1), name=name+"_Branch_2_2")

    branch_3 = MaxPooling2D( (3, 3), strides=(1, 1), padding='same', name=name+"_Branch_3")(x)
    branch_3 = Conv2d_bn(branch_3, br3[0], (1, 1), name=name+"_Branch_3_1")

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name=name+"_Concatenated")
    return x
img_input = Input(shape=(224,224,1))
x=InceptionV3_block(img_input, ((64,), (96,128), (16,32), (32,)), 3, 'Block_1')
print(x)


def VGG16_Inception(include_top=True, input_tensor=None, input_shape=(224, 224, 1),
                    pooling='max', classes=1000):
    img_input = Input(shape=input_shape)

    x = Conv2d_bn(img_input, 64, (3, 3), activation='relu', padding='same', name='block1_conv1')
    x = Conv2d_bn(x, 64, (3, 3), activation='relu', padding='same', name='block1_conv2')
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2d_bn(x, 128, (3, 3), activation='relu', padding='same', name='block2_conv1')
    x = Conv2d_bn(x, 128, (3, 3), activation='relu', padding='same', name='block2_conv2')
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = InceptionV1_block(x, ((64,), (96, 128), (16, 32), (32,)), 3, 'Block_3_1')
    x = InceptionV1_block(x, ((64,), (96, 128), (16, 32), (32,)), 3, 'Block_3_2')
    x = InceptionV1_block(x, ((64,), (96, 128), (16, 32), (32,)), 3, 'Block_3_3')
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2d_bn(x, 512, (3, 3), activation='relu', padding='same', name='block4_conv1')
    x = Conv2d_bn(x, 512, (3, 3), activation='relu', padding='same', name='block4_conv2')
    x = Conv2d_bn(x, 512, (3, 3), activation='relu', padding='same', name='block4_conv3')
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = InceptionV3_block(x, ((128,), (192, 256), (32, 64), (64,)), 3, 'Block_5_1')
    x = InceptionV3_block(x, ((128,), (192, 256), (32, 64), (64,)), 3, 'Block_5_2')
    x = InceptionV3_block(x, ((128,), (192, 256), (32, 64), (64,)), 3, 'Block_5_3')
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    inputs = img_input
    # Create model.
    model = Model(inputs, x, name='vgg16')

    return model
model = VGG16_Inception(include_top=False)
model.summary()
