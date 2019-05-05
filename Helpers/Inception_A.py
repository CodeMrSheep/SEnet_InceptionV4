"""
Created on Sun May  5 21:43:18 2019

@author: nour
"""
from InceptionV4_SEnet import Conv2dBn
from keras.layers import concatenate, AveragePooling2D

def Inception_A(x, channel_axis):
    split_X1 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    split_X1 = Conv2dBn(split_X1, 96, 1, 1)
                
    split_X2 = Conv2dBn(x, 96, 1, 1)
            
    split_X3 = Conv2dBn(x, 64, 1, 1)
    split_X3 = Conv2dBn(split_X3, 96, 3, 3)
            
    split_X4 = Conv2dBn(x, 64, 1, 1)
    split_X4 = Conv2dBn(split_X4, 96, 3, 3)
    split_X4 = Conv2dBn(split_X4, 96, 3, 3)
            
    x = concatenate([split_X1, split_X2, split_X3, split_X4], axis= channel_axis)
                        
    return x