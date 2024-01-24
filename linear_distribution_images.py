import os
import random
import shutil
from collections import defaultdict
import re
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
from PIL import Image

def delete_photos_linear(folder):
    source_images_dir = os.path.join(folder, 'UTKFace')
    all_images = os.listdir(source_images_dir)
    random.shuffle(all_images)

    my_dict_ages = defaultdict(int)

    for i, image in enumerate(all_images):
        # print(image)
        image_age = int(image.split("_")[0])//10
        if (image_age >= 7):
            image_age = 7
        my_dict_ages[image_age] += 1

    print(my_dict_ages)

    min_photos = min(my_dict_ages.values())
    my_dict_ages_new = defaultdict(int)

    for i, image in enumerate(all_images):
        image_age = int(image.split("_")[0])//10
        if (image_age >= 7):
            image_age = 7
    
        print(i)
    
        my_dict_ages_new[image_age] += 1
        if (my_dict_ages_new[image_age] > min_photos):
            image_file_name = os.path.join("UTKFace", image)
            if os.path.isfile(image_file_name):
                # Delete the file
                os.remove(image_file_name)
            else:
                print("Error: %s file not found" % image)

if __name__ == '__main__':
    baseDir = 'C:\\Users\\User\\Desktop\\Szkoła\\MachineLearning\\AgeRecognition'
    delete_photos_linear(baseDir)