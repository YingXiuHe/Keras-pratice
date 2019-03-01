#-*-coding: utf-8-*-

import keras 
from keras.layers import Flatten, Conv2D, MaxPool2D, Dropout, Activation, Dense
from keras.models import Sequential, Model
from keras.utils import plot_model
from IPython.display import Image

input_shape = (227,227,3)
model = Sequential(name='AlexNet')

model.add(Conv2D(96, (11,11), strides=(4,4), activation='relu',padding='valid', input_shape=input_shape, kernel_initializer='uniform'))
model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 3)))

model.add(Conv2D(256, (5, 5), strides=(1, 1),activation='relu', padding='same', kernel_initializer='uniform'))
model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))

model.add(Conv2D(384, (3,3), strides=(1,1), activation='relu', padding='same', kernel_initializer='uniform'))
model.add(Conv2D(384, (3,3), strides=(1,1), activation='relu', padding='same', kernel_initializer='uniform'))
model.add(Conv2D(256, (3,3), strides=(1,1), activation='relu', padding='same', kernel_initializer='uniform'))
model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))

model.add(Flatten(name='Flatten_layer'))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000, activation='softmax', name='predicts'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['acc'])

plot_model(model, to_file='AlexNet.png')
Image('AlexNet.png')
