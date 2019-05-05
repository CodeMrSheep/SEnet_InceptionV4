"""
Created on Sun May  5 21:52:03 2019

@author: nour
"""

from InceptionV4_SEnet import Conv2dBn
from keras.layers import concatenate, MaxPooling2D

def Reduction_B(x, channel_axis):
    splitMax_X1 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)
        
    split_X2 = Conv2dBn(x, 256, 1, 1)
    split_X2 = Conv2dBn(split_X2, 384, 3, 3, strides=(2, 2), padding='valid')
        
    split_X3 = Conv2dBn(x, 256, 1, 1)
    split_X3 = Conv2dBn(split_X3, 288, 3, 3, strides=(2, 2), padding='valid')
    
    split_X4 = Conv2dBn(x, 256, 1, 1)
    split_X4 = Conv2dBn(split_X4, 288, 3, 3)
    split_X4 = Conv2dBn(split_X4, 320, 3, 3, strides=(2, 2), padding='valid')
        
    x = concatenate([splitMax_X1, split_X2, split_X3, split_X4], axis= channel_axis)
    
    return x