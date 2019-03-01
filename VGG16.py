#-*-coding: utf-8-*-

#用sequential模块构建vgg16
import platform
import tensorflow
import keras
from IPython.display import Image
from keras.layers import Conv2D, MaxPool2D
from keras.layers import Flatten, Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import plot_model

# 创建模型
input_shape = (224, 224, 3)
model = Sequential(name='vgg16')

#第一个卷积块
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape, name='block1_conv1'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
model.add(MaxPool2D((2, 2), strides=(2, 2), name='block1_pool'))

#第二个卷积块
model.add(Conv2D(128,(3, 3), activation='relu', padding='same', name='block2_conv1'))
model.add(Conv2D(128,(3, 3), activation='relu', padding='same', name='block2_conv2'))
model.add(Conv2D(128,(3, 3), activation='relu', padding='same', name='block2_conv3'))
model.add(MaxPool2D((2, 2), strides=(2, 2), name='block2_pool'))
          
#第三个卷积块
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
model.add(MaxPool2D((2, 2), strides=(2, 2), name='block3_pool'))

#第四个卷积块
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
model.add(MaxPool2D((2, 2), strides=(2, 2), name='block4_pool'))
          
#第五个卷积块
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
model.add(MaxPool2D((2, 2), strides=(2, 2), name='block5_pool'))
          
model.add(Flatten(name='flatten'))
model.add(Dense(4096, activation='relu', name='fc1'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu', name='fc2'))
model.add(Dense(1000, activation='softmax', name='predictions'))

model.summary()

plot_model(model, to_file='vgg16.png')
Image('vgg16.png')

'''
#用keras function-api来创建模型
input_shape = (224,224,3)

#第一个block卷积块
img_input = Input(shape=input_shape, name='input')
x = Conv2D(64, (3,3), activation='relu', padding='same', name='block1_conv1')(img_input)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
x = MaxPool2D((2, 2), strides=(2, 2), name='block1_pool')(x)

#第二个卷积块
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv3')(x)
x = MaxPool2D((2, 2), strides=(2, 2), name='block2_pool')(x)

#第三个卷积块
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
x = MaxPool2D((2, 2), strides=(2, 2), name='block3_pool')(x)

#第四个卷积块
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
x = MaxPool2D((2, 2), strides=(2, 2), name='block4_pool')(x)

#第五个卷积块
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
x = MaxPool2D((2, 2), strides=(2, 2), name='block5_pool')(x)

x = Flatten(name='flatten')(x)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dropout(0.5, name='drop_layer')(x)
x= Dense(4096, activation='relu', name='fc2')(x)
output = Dense(1000, activation='softmax', name='predtctions')(x)

model = Model(inputs=img_input, outputs=output)
model.summary()
plot_model(model, to_file='vgg.png')
Image('vgg.png')
'''