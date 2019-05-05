"""
Created on Fri Apr 12 22:04:44 2019

@author: nour
"""

import numpy as np
import os
from PIL import Image

def LoadDataset():
    data_dir = '/data'
    classes_to_load = ['Manhole', 'Bump', 'Pothole', 'Rumble Strip']
    
    classes = {
      'Manhole': 1,
      'Bump': 2,
      'Pothole': 3,
      'Rumble Strip': 4
    }
    
    images_list = list()
    labels_list = list()

    for single_class in classes_to_load:
        image_train_files_dir = list(filter(lambda x: x[0] != '.', os.listdir(data_dir + '/train/' + single_class)))
        image_test_files_dir = list(filter(lambda x: x[0] != '.', os.listdir(data_dir + '/test/' + single_class)))
        
        for img_path in image_train_files_dir:
            identifierIndex = img_path.find('.') + 1
            identifier = img_path[identifierIndex:]
            if identifier == 'png':
                full_path = data_dir + '/train/' + single_class + '/' + img_path
                images_list.append(np.array(Image.open(full_path).resize((256, 256), Image.ANTIALIAS)))
                arr = [0, 0, 0, 0]
                arr[classes[single_class] - 1] = 1
                labels_list.append(arr)
            
        for img_path in image_test_files_dir:
            identifierIndex = img_path.find('.') + 1
            identifier = img_path[identifierIndex:]
            if identifier == 'png':
                full_path = data_dir + '/test/' + single_class + '/' + img_path
                images_list.append(np.array(Image.open(full_path).resize((256, 256), Image.ANTIALIAS)))
                arr = [0, 0, 0, 0]
                arr[classes[single_class] - 1] = 1
                labels_list.append(arr)
            
    return images_list, labels_list
