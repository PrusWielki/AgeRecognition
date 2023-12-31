import sys
import os
import shutil
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import time

current_directory = os.getcwd()

def __main__():
    if len(sys.argv) != 4:
        print(f"length: {len(sys.argv)}, argv: {sys.argv}")
        usage()
        return
    
    # Load the pre-trained model
    model_name = sys.argv[1]
    model_path = os.path.join(current_directory, model_name)
    model = load_model(model_path)

    # Create directory or remove its content to save data
    directory_name = sys.argv[2]
    directory_path = os.path.join(current_directory, directory_name)
    create_directory(directory_path)

    images_path = sys.argv[3]
    iterate_images(images_path, model, directory_path)

def iterate_images(directory_path, model, eval_dir):
    # Ensure the directory exists
    if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
        print(f"The specified path '{directory_path}' is not a valid directory.")
        return
    
    # List all files in the directory
    files = os.listdir(directory_path)
    
    # Filter out only image files (e.g., JPEG, PNG)
    image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No image files found in '{directory_path}'.")
        return
    
    for image_file in image_files:
        try:
            # print(f"Processing image: {image_file}")
            eval_rashi_dataset(image_file, directory_path, model, eval_dir)
            # print(f"Processed image: {image_file}")
        except Exception as e:
            print(f"Error processing {image_file}. Reason: {e}")

# specific logic for data from: https://www.kaggle.com/datasets/rashikrahmanpritom/age-recognition-dataset
# or other images that contains labelled age in the format: AGE_x_y_z.jpg (age must be in the first index separated by '_' sign)
def eval_rashi_dataset(image_file, dir_path, model, eval_dir):
    splitted_text = image_file.split('_')
    labelled_age = splitted_text[0]
    # print(f"file_name: {image_file}, labelled_age: {labelled_age}")
    img = get_image_features(os.path.join(dir_path, image_file))
    predictions = model.predict(img)
    age = round(predictions[0][0])
    difference = int(labelled_age) - age;
    write_to_text_file(eval_dir, 'evaluation.txt', f'Model:{age}, Label:{labelled_age}, Difference:{str(difference)}')

def write_to_text_file(dir_path, file_name, content):
    file_path = os.path.join(dir_path, file_name)

    if os.path.exists(file_path):
        with open(file_path, 'a') as file:
            file.write(content + '\n')
    else:
        with open(file_path, 'w') as file:
            file.write(content + '\n')
    

def get_image_features(image_path):
    img = image.load_img(image_path, grayscale=True)
    img = img.resize((128, 128), Image.LANCZOS)
    img = image.img_to_array(img)
    img = img.reshape(1, 128, 128, 1)
    img = img / 255.0
    return img

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory at path: {path}")
    else:
        print("Directory already exists, clearing content...")
        # Clear the contents of the directory
        for file_name in os.listdir(path):
            file_path = os.path.join(path, file_name)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

def usage():
    print("Proper usage is: python model-eval.py [model_name] [directory_name_to_save_files] [path_to_eval_images]")
    print("[model_name] - name of the model you want to use to evaluate your data")
    print("[directory_name_to_save_files] - name of the directory in the current directory which will be created and .txt file with results will be saved into")
    print("[path_to_eval_images] - absolute path to the directory containing images to be evaluated")

if __name__ == "__main__":
    __main__()
