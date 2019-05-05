"""
Created on Sun May  5 22:08:41 2019

@author: nour
"""

from InceptionV4_SEnet import Conv2dBn
from keras.layers import concatenate, MaxPooling2D

def Stem(x, channel_axis):
    x = Conv2dBn(x, 32, 3, 3, padding='valid')
    x = Conv2dBn(x, 32, 3, 3, strides=(2, 2), padding='valid')
    x = Conv2dBn(x, 64, 3, 3)
        
    splitMax_X = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)
    splitConv_X= Conv2dBn(x, 96, 3, 3, strides=(2, 2), padding='valid')
        
    x = concatenate([splitMax_X, splitConv_X], axis= channel_axis)
        
    split_X1 = Conv2dBn(x, 64, 1, 1)
    split_X1 = Conv2dBn(split_X1, 96, 3, 3, padding='valid')
        
    split_X2 = Conv2dBn(x, 64, 1, 1)
    split_X2 = Conv2dBn(split_X2, 64, 7, 1)
    split_X2 = Conv2dBn(split_X2, 64, 1, 7)
    split_X2 = Conv2dBn(split_X2, 96, 3, 3, padding='valid')
        
    x = concatenate([split_X1, split_X2], axis= 3)
        
    splitMax_X = MaxPooling2D(strides=(2, 2), padding='valid')(x)
    splitConv_X= Conv2dBn(x, 192, 3, 3, padding='valid')
        
    x = concatenate([splitMax_X, splitConv_X], axis= channel_axis)
        
    return x