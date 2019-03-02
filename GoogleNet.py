#-*-coding: utf-8-*-
import keras 
from keras.layers import *
from keras.utils import plot_model
from keras.models import Sequential, Model
from IPython.display import Image


def Pridect(x):
    x = AveragePooling2D(pool_size=(5,5), strides=(3,3), padding='same')(x)
    x = Conv2D(128, kernel_size=(1,1), strides=(1,1), padding='same', activation='relu')(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(classes, activation='softmax')(x)

    return x

def Inception(x, filters):
    branch1 = Conv2D(filters[0], kernel_size=(1,1), strides=(1, 1), padding='same')(x)

    branch2 = Conv2D(filters[1], kernel_size=(1,1), strides=(1, 1), padding='same')(x)
    branch2_1 = Conv2D(filters[2], kernel_size=(3,3), strides=(1, 1), padding='same')(branch2)    

    branch3 = Conv2D(filters[3], kernel_size=(1,1), strides=(1,1), padding='same')(x)
    branch3_1 = Conv2D(filters[4], kernel_size=(5, 5), strides=(1,1), padding='same')(branch3)

    branch4 = MaxPool2D(pool_size=(3,3), strides=(1,1), padding='same')(x)
    branch4_1 = Conv2D(filters[5], kernel_size=(1,1), strides=(1, 1), padding='same')(branch4)

    x = concatenate([branch1, branch2_1, branch3_1, branch4_1])

    return x


input = Input(shape=(224, 224, 3))
classes = 1000

x = Conv2D(64, kernel_size=(7,7), strides=(2,2), padding='same', activation='relu')(input)
x = MaxPool2D(pool_size=(3,3), strides=(2,2))(x)
x = BatchNormalization(axis=3)(x)
x = Conv2D(64, kernel_size=(1,1), strides=(1,1), padding='valid', activation='relu')(x)
x = Conv2D(192, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')(x)

x = Inception(x, filters=[64, 96, 128, 16, 32, 32])
x = Inception(x, filters=[128, 128, 192, 32, 96, 64]) 
x = MaxPool2D(pool_size=(3,3), strides=(2,2))(x)
x = Inception(x, filters=[192, 96, 208, 16, 48, 64])
x = Inception(x, filters=[160, 112, 224, 24, 64, 64])
output1 = Pridect(x)

x = Inception(x, filters=[128,128,256,24,64,64])
x = Inception(x, filters=[112,144,288,32,64,64])
output2 = Pridect(x)

x = MaxPool2D(pool_size=(3,3), strides=(2,2))(x)
x = Inception(x, filters=[256,160,320,32,128,128])
x = Inception(x, filters=[384,192,384,48,128,128])
x = AvgPool2D(pool_size=(7,7), strides=(1,1))(x)

x = Flatten()(x)
x = Dropout(0.5)(x)
output3 = Dense(classes, activation='softmax',name='pridects')(x)

model = Model(inputs=input, outputs=[output1, output2, output3])
model.summary()
    
plot_model(model=model, to_file='GoogleNet.png')
Image('GoogleNet.png')
