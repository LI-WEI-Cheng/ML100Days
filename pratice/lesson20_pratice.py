# Transfer Learning:
# 為深度學習中常見的做法，其概念是利用過去訓練過的結構與學習到的權重來加快這次的訓練任務。
# 而之所以能夠這麼做是因為不同的分類任務間仍有許多共享的特徵，如類別貓與花瓶間也可能有相似的輪廓，
# 因此過去訓練過分類貓的分類器，其權重也能使用來加速這次訓練花瓶分類器的速度。
# 通常淺層的卷積核學到比較粗略、泛用的特徵，因此在做 Transfer Learning 時，當新的 Dataset 與原本的 Dataset 相近時，
# 可以考慮不更新淺層 Kernel 的權重(freeze)，又或是資料不足，擔心 Overfitting 時也可以使用。
# Freeze 的另一個好處是能加快訓練，然而當訓練資料的特徵差異太大時，
# 還是建議可以全部開啟或只 freeze 少數層，而要 freeze 具體的層數，並沒有一定的規範。
# 具體實作第一步，我們可以從 Keras 讀入特定模型架構其中 weight= ‘imagenet’ 代表我們要輸入 ImageNet 的 pretrained weight，
# 用來做 Transfer Learning。
# 第二步：可以自己在原本架構後面再新增幾層，尤其當沒有導入Fully connected layers時，是必要自己加上 FC 做分類。
# 第三步：定義要 Freeze 的層數，之後就跟訓練一般模型一樣了。
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import keras
from keras.layers import Input

from keras.datasets import cifar10
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

input_tensor = Input(shape=(32, 32, 3))
# include top 決定要不要加入 fully Connected Layer

'''Xception 架構'''
# model=keras.applications.xception.Xception(include_top=False, weights='imagenet',
# input_tensor=input_tensor,pooling=None, classes=2)
'''Resnet 50 架構'''
model = keras.applications.ResNet50(input_shape=(32, 32, 3), include_top=False, weights='imagenet', pooling=None,
                                    classes=10)
model.summary()
x = model.output
x = GlobalAveragePooling2D()(x)
x = Dense(output_dim=128, activation='relu')(x)
x=Dropout(p=0.1)(x)
predictions = Dense(output_dim=10,activation='softmax')(x)
model = Model(inputs=model.input, outputs=predictions)
print('Model深度：', len(model.layers))

for layer in model.layers[:100]:
    layer.trainable = False
for layer in model.layers[100:]:
    layer.trainable = True
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape)  # (50000, 32, 32, 3)


## Normalize Data
def normalize(X_train, X_test):
    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)
    return X_train, X_test


## Normalize Training and Testset
x_train, x_test = normalize(x_train, x_test)

## OneHot Label 由(None, 1)-(None, 10)
## ex. label=2,變成[0,0,1,0,0,0,0,0,0,0]
one_hot = OneHotEncoder()
y_train = one_hot.fit_transform(y_train).toarray()
y_test = one_hot.transform(y_test).toarray()


