import os
import random
import shutil
from collections import defaultdict
import re
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
from PIL import Image

def augmentate_data(folder):
    source_images_dir = os.path.join(folder, 'UTKFace')
    all_images = os.listdir(source_images_dir)
    datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.2
    )
    save_to_dir = 'C:\\Users\\kacpe\\OneDrive\\Pulpit\\Uczelnia\\semestr5\\MachineLearning\\UTKFace\\AugmentedUTKFace'
    # Check if the directory exists
    if os.path.exists(save_to_dir):
        # Remove the directory and all its contents
        shutil.rmtree(save_to_dir)

    # Recreate the directory
    os.makedirs(save_to_dir)


    # Loop over all images
    for i, image in enumerate(all_images):
        # Extract the age from the filename
        age = re.findall(r'^\d+', image)[0]

        # Load the image
        img = load_img(os.path.join(source_images_dir, image))
        img_array = img_to_array(img)
        img_array = img_array.reshape((1,) + img_array.shape)  # Add batch dimension
        print(i)

        # Generate 7 augmented images
        for i in range(7):
            # Apply the transformations
            img_augmented = next(datagen.flow(img_array, batch_size=1))
            img_augmented = img_augmented[0]

            # Save the augmented image
            img_augmented = Image.fromarray((img_augmented * 255).astype(np.uint8))
            img_augmented.save(os.path.join(save_to_dir, f'{age}_augmented_{i}_{image}'))

def delete_photos_linear(folder):
    source_images_dir = os.path.join(folder, 'AugmentedUTKFace')
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
            image_file_name = os.path.join("UTKFace\\AugmentedUTKFace", image)
            if os.path.isfile(image_file_name):
                # Delete the file
                os.remove(image_file_name)
            else:
                print("Error: %s file not found" % image)


if __name__ == "__main__":
    # augmentate_data("C:\\Users\\kacpe\\OneDrive\\Pulpit\\Uczelnia\\semestr5\\MachineLearning\\UTKFace")
    delete_photos_linear('C:\\Users\\kacpe\\OneDrive\\Pulpit\\Uczelnia\\semestr5\\MachineLearning\\UTKFace')
