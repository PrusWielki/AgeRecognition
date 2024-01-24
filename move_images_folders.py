import os
import shutil
import random

def move_images(folder, folderUTK):
    source_images_dir = os.path.join(folder, str(folderUTK))

    train_images_dir = os.path.join(folder, 'Train')
    test_images_dir = os.path.join(folder, 'Test')
    val_images_dir = os.path.join(folder, 'Val')

    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(test_images_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    all_images = os.listdir(source_images_dir)
    random.shuffle(all_images)

    total_images = len(all_images)
    train_split = int(0.7 * total_images)
    test_split = int(0.15 * total_images)
    val_split = total_images - train_split - test_split

    for i, image in enumerate(all_images):
        if i < train_split:
            shutil.move(os.path.join(source_images_dir, image), os.path.join(train_images_dir, image))
        elif i < train_split + test_split:
            shutil.move(os.path.join(source_images_dir, image), os.path.join(test_images_dir, image))
        else:
            shutil.move(os.path.join(source_images_dir, image), os.path.join(val_images_dir, image))