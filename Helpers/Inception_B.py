"""
Created on Sun May  5 21:50:12 2019

@author: nour
"""

from InceptionV4_SEnet import Conv2dBn
from keras.layers import concatenate, AveragePooling2D

def Inception_B(x, channel_axis):
    split_X1 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    split_X1 = Conv2dBn(split_X1, 128, 1, 1)
            
    split_X2 = Conv2dBn(x, 384, 1, 1)
            
    split_X3 = Conv2dBn(x, 192, 1, 1)
    split_X3 = Conv2dBn(split_X3, 224, 1, 7)
    split_X3 = Conv2dBn(split_X3, 256, 1, 7)
            
    split_X4 = Conv2dBn(x, 192, 1, 1)
    split_X4 = Conv2dBn(split_X4, 192, 1, 7)
    split_X4 = Conv2dBn(split_X4, 224, 7, 1)
    split_X4 = Conv2dBn(split_X4, 224, 1, 7)
    split_X4 = Conv2dBn(split_X4, 256, 7, 1)
            
    x = concatenate([split_X1, split_X2, split_X3, split_X4], axis= channel_axis)
                        
    return x