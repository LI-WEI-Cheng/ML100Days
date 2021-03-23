from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import BatchNormalization
# ------------------------------------------------------------------------------------------------------------------- #
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# ------------------------------------------------------------------------------------------------------------------- #
classifier=Sequential()
input = (32, 32, 3)
classifier.add(Convolution2D(32, kernel_size=(3, 3), padding='same',input_shape=input))
classifier.add(BatchNormalization())
classifier.add(Activation('relu'))
classifier.summary()