from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Input, Dense
from keras.layers import Activation, Dense
classifier=Sequential()
classifier.add(Convolution2D(
    batch_input_shape=(None, 1, 28, 28),
    filters=32,
    kernel_size=5,
    strides=1,
    padding='same',     # Padding method
    data_format='channels_first',
))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding='same',
    data_format='channels_first',
))
# model = Model(inputs=inputs, outputs=x)
classifier.summary()
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