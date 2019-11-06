#!/usr/bin/env python 
# keras imports 

import keras.models as models
from keras.layers.core import Reshape, Dense, Dropout, Activation, Flatten
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.convolutional import Conv1D, MaxPooling1D, Conv2D
from keras.optimizers import adam
import tensorflow as t



def gvn_00(n_class, in_shp):
    model = models.Sequential()
    model.add(Conv1D(filters=128,
                 kernel_size=16,
                 strides=1,
                 padding='valid', 
                 activation="relu", 
                 data_format='channels_first',
                 name="conv1", 
                 kernel_initializer='glorot_uniform',
                 input_shape=in_shp))
    model.add(Dropout(rate=.5))
    model.add(MaxPooling1D(pool_size=1,padding='valid', name="pool1"))
    model.add(Conv1D(filters=128,
                 kernel_size=8,
                 strides=4,
                 padding='valid', 
                 activation="relu", 
                 data_format='channels_first',
                 name="conv2", 
                 kernel_initializer='glorot_uniform'))
    model.add(Dropout(rate=.5))
    model.add(MaxPooling1D(pool_size=1, padding='valid', name="pool2"))
    model.add(Conv1D(filters=128,
                 kernel_size=2,
                 strides=1,
                 padding='valid', 
                 activation="relu", 
                 name="conv3", 
                 data_format='channels_first',
                 kernel_initializer='glorot_uniform'))
    model.add(Dropout(rate=.5))
    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_initializer='he_normal', name="dense1"))
    model.add(Dropout(rate=.5))
    model.add(Dense(n_class, kernel_initializer='he_normal', name="dense2" ))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


