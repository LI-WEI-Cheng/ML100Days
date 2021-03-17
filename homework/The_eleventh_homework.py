from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import Input, Dense
from keras.models import Model

classifier=Sequential()
classifier.add(Convolution2D(32,3,3,input_shape=(28,28,1)))

print(classifier.summary())
classifier=Sequential()
inputs = Input(shape=(784,))
x=Dense(288)(inputs)
model = Model(inputs=inputs, outputs=x)
print(model.summary())