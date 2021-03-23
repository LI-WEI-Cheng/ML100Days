from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Input,Dense
from keras.layers import GlobalAveragePooling2D
import os   # 去除bug
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
input_shape = (32, 32, 3)
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), padding='same',input_shape=input_shape))
model.add(MaxPooling2D())

model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
model.add(MaxPooling2D())

model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
model.add(MaxPooling2D())

model.add(Conv2D(10, kernel_size=(3, 3), padding='same'))
model.add(Flatten())

model.add(Dense(288))
model.summary()
print('--------------------------------------------------------------------------------------------------------------')
