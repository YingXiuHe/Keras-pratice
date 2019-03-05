#-*-coding:utf-8-*-

import keras
import keras.backend as K 
from keras import Model, Input
from keras.layers import Conv2D, Dense, Dropout, Activation, Concatenate
from keras.layers.pooling import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

from keras.utils import plot_model
from IPython.display import Image

'''
参考链接:https://github.com/tdeboissiere/DeepLearningImplementations/blob/master/DenseNet/densenet.py
'''


def conv_block(input, nb_filter, dropout_rate=None, weight_decay=1e-4):
    x = Activation(activation='relu')(input)
    x = Conv2D(nb_filter, kernel_size=(3, 3), padding='same', use_bias=False, kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    return x

def dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4):
    if K.image_dim_ordering() == 'th':
        concat_axis = 1
    elif K.image_dim_ordering() == 'tf':
        concat_axis = -1
    
    feature_list = [x]

    for i in range(nb_layers):
        x = conv_block(x, growth_rate, dropout_rate, weight_decay)
        feature_list.append(x)
        x = Concatenate(axis=concat_axis)(feature_list)
        nb_filter += growth_rate
    
    return x, nb_filter


def transition(input, nb_filter, dropout_rate=None, weight_decay=1e-4):
    if K.image_dim_ordering() == 'th':
        concat_axis = 1
    elif K.image_dim_ordering() == 'tf':
        concat_axis = -1
    x = Conv2D(nb_filter, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False,
                kernel_initializer='he_uniform', kernel_regularizer=l2(weight_decay))(input)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)

    return x


'''
def DensNet(classes=10, nb_filter=16, img_dim=(32, 32, 3), depth=40, nb_dense_block=3, dropout_rate=0.5, weight_decay=1E-4, growth_rate=12):
    model_input = Input(shape=img_dim)
    concat_axis = 1 if K.image_dim_ordering() == 'th' else -1
    #assert (depth - 4) %3 == 0,
    nb_layers = int((depth-4) / 3)

    x  = Conv2D(nb_filter, kernel_size=(3, 3), strides=1, padding='same', use_bias=False, kernel_regularizer=l2(weight_decay))(model_input)
    
    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)

    for block_idx in range(nb_dense_block - 1):
        x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)
        x = transition(x, nb_filter, dropout_rate=dropout_rate, weight_decay=weight_decay)

    x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate,dropout_rate=dropout_rate, weight_decay=weight_decay)
    x = Activation(activation='relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(classes, activation='softmax', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(x)

    model = Model(inputs=model_input, outputs=x)
    model.summary()
    plot_model(model=model, to_file='DenseNet.png')
    Image('DenseNet.png')
    return model, plot_model
'''
classes=10
nb_filter=16
img_dim=(32, 32, 3)
depth=40
nb_dense_block=3
dropout_rate=0.5
weight_decay=1e-4
growth_rate=12    

model_input = Input(shape=img_dim)
concat_axis = 1 if K.image_dim_ordering() == 'th' else -1
assert (depth - 4) % 3 == 0
nb_layers = int((depth-4) / 3)

x  = Conv2D(nb_filter, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, kernel_regularizer=l2(weight_decay))(model_input)
    
x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)

for block_idx in range(nb_dense_block - 1):
#for block_idx in range(2):

    x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)
    x = transition(x, nb_filter, dropout_rate=dropout_rate, weight_decay=weight_decay)

x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)
x = Activation(activation='relu')(x)
x = GlobalAveragePooling2D()(x)
x = Dense(classes, activation='softmax', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(x)

model = Model(inputs=model_input, outputs=x)
model_variables = model.summary()
plot_model(model=model, to_file='DenseNet.png')
Image('DenseNet.png')



