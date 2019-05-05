"""
Created on Sun May  5 21:53:35 2019

@author: nour
"""

from InceptionV4_SEnet import Conv2dBn
from keras.layers import concatenate, AveragePooling2D

def Inception_C(x, channel_axis):
    split_X1 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    split_X1 = Conv2dBn(split_X1, 256, 1, 1)
            
    split_X2 = Conv2dBn(x, 256, 1, 1)
            
    split_X3 = Conv2dBn(x, 384, 1, 1)
    split_X3_1 = Conv2dBn(split_X3, 256, 1, 3)
    split_X3_2 = Conv2dBn(split_X3, 256, 3, 1)
            
    split_X4 = Conv2dBn(x, 384, 1, 1)
    split_X4 = Conv2dBn(split_X4, 448, 1, 3)
    split_X4 = Conv2dBn(split_X4, 512, 3, 1)
    split_X4_1 = Conv2dBn(split_X4, 256, 3, 1)
    split_X4_2 = Conv2dBn(split_X4, 256, 1, 3)
            
    x = concatenate([split_X1, split_X2, split_X3_1, split_X3_2, split_X4_1, split_X4_2], axis= channel_axis)
                    
    return x