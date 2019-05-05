"""
Created on Sun May  5 21:47:21 2019

@author: nour
"""

from InceptionV4_SEnet import Conv2dBn
from keras.layers import concatenate, MaxPooling2D

def Reduction_A(x, channel_axis):
    l = 256
    k = 256
    m = 384
    n = 384
        
    splitMax_X1 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)
        
    split_X2 = Conv2dBn(x, n, 3, 3, strides=(2, 2), padding='valid')
        
    split_X3 = Conv2dBn(x, k, 1, 1)
    split_X3 = Conv2dBn(split_X3, l, 3, 3)
    split_X3 = Conv2dBn(split_X3, m, 3, 3, strides=(2, 2), padding='valid')
        
    x = concatenate([splitMax_X1, split_X2, split_X3], axis= channel_axis)
        
    return x