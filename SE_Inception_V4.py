"""
Created on Sat May  4 20:54:34 2019

@author: nour
"""

from __future__ import print_function
from __future__ import absolute_import

from keras.models import Model
from keras.layers import Activation, Dense, Dropout, Input, BatchNormalization, Conv2D, GlobalAveragePooling2D
from keras.engine.topology import get_source_inputs
from keras import backend as K
from keras_applications.imagenet_utils import _obtain_input_shape

from Helpers.SqueezeAndExecute import SqueezeAndExecute
from Helpers.Stem import Stem
from Helpers.Inception_A import Inception_A
from Helpers.Inception_B import Inception_B
from Helpers.Inception_C import Inception_C
from Helpers.Reduction_A import Reduction_A
from Helpers.Reduction_B import Reduction_B

def Conv2dBn(x, filters, numRow, numCol, padding='same', strides=(1, 1), name=None):

    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
        
    x = Conv2D(filters, (numRow, numCol), strides=strides, padding=padding, use_bias=False)(x)
    x = BatchNormalization(axis=bn_axis, scale=False)(x)
    x = Activation('relu', name=name)(x)
    return x


def SEInceptionV4(include_top=True, weights=None, input_tensor=None, input_shape=None, classes=4):

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape, default_size=299, min_size=139, data_format=K.image_data_format(), require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    x = Stem(input_shape, channel_axis)

    for i in range(4) :
        x = Inception_A(x, channel_axis)
        x = SqueezeAndExecute(x)
    x = Reduction_A(x, channel_axis)

    for i in range(7)  :
        x = Inception_B(x, channel_axis)
        x = SqueezeAndExecute(x)
    x = Reduction_B(x, channel_axis)

    for i in range(3) :
        x = Inception_C(x, channel_axis)
        x = SqueezeAndExecute(x)

    if include_top:
        # Classification block
        x = GlobalAveragePooling2D()(x)
        x = Dropout(x, rate=0.2)
        x = Dense(classes, activation='softmax')(x)

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='inception_v4')

    return model