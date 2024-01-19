import sys
import os
import shutil
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import time

current_directory = os.getcwd()

# global dict for age groups
age_ranges_dict = {
    '6-10': 0.0,
    '11-20': 0.0,
    '21-30': 0.0,
    '31-40': 0.0,
    '41-50': 0.0,
    '51-60': 0.0,
    '61-70': 0.0,
    '71-80': 0.0,
    '81-90': 0.0,
    '91-100': 0.0
}

def __main__():
    if len(sys.argv) != 5:
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
    eval_file_name = 'evaluation.txt'

    images_path = sys.argv[3]
    accuracy_param = int(sys.argv[4])
    iterate_images(images_path, model, directory_path, accuracy_param)
    read_and_prepare_eval_results(directory_path, eval_file_name, 'result.txt', accuracy_param)

def iterate_images(directory_path, model, eval_dir, accuracy_param):
    if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
        print(f"The specified path '{directory_path}' is not a valid directory.")
        return
    
    files = os.listdir(directory_path)
    
    image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No image files found in '{directory_path}'.")
        return
    
    for image_file in image_files:
        try:
            eval_rashi_dataset(image_file, directory_path, model, eval_dir, accuracy_param)
        except Exception as e:
            print(f"Error processing {image_file}. Reason: {e}")

# specific logic for data from: https://www.kaggle.com/datasets/rashikrahmanpritom/age-recognition-dataset
# or other images that contains labelled age in the format: AGE_x_y_z.jpg (age must be in the first index separated by '_' sign)
def eval_rashi_dataset(image_file, dir_path, model, eval_dir, accuracy_param):
    img = get_image_features(os.path.join(dir_path, image_file))

    predictions = model.predict(img)
    age = round(predictions[0][0])

    splitted_text = image_file.split('_')
    labelled_age = splitted_text[0]
    difference = abs(int(labelled_age) - age);
    percent_difference = abs(int(difference)) / int(labelled_age) * 100
    
    eval_file_name = 'evaluation.txt'
    write_to_text_file(eval_dir, eval_file_name, f'| Model:{age:3} | Label:{labelled_age:3} | Difference:{difference} | PercentageDifference:{str(round(percent_difference)):3} | Accurate:{round(percent_difference) <= accuracy_param:1} |')

def write_to_text_file(dir_path, file_name, content):
    file_path = os.path.join(dir_path, file_name)

    if os.path.exists(file_path):
        with open(file_path, 'a') as file:
            file.write(content + '\n')
    else:
        with open(file_path, 'w') as file:
            file.write(content + '\n')

# read a file in a specific format and calculate errors
def read_and_prepare_eval_results(eval_dir, data_file_name, eval_file_name, accuracy_param):
    data_file = os.path.join(eval_dir, data_file_name)
    if os.path.exists(data_file):
        sum_diff = 0
        sum_diff_percentage = 0
        accuracy = 0
        counter = 0
        with open(data_file, 'r') as file:
            for line in file:
                pairs = line.split('|')
                values = {}
                for pair in pairs:
                    if ':' in pair:
                        key, value = pair.split(':', 1)  # Specify the maximum number of splits
                        # print(f'key: {key}, value: {value}')
                        values[key.strip()] = int(value.strip()) if value.strip().isdigit() else value.strip()

                if 'Difference' in values:
                    sum_diff += int(values['Difference'])

                if 'PercentageDifference' in values and isinstance(values['PercentageDifference'], int):
                    sum_diff_percentage += values['PercentageDifference']

                counter += 1

                if 'Accurate' in values and values['Accurate'] == 1:
                    accuracy += 1

        average_difference = sum_diff / counter if counter > 0 else 0
        avg_difference_percentage = float(sum_diff_percentage / counter) if counter > 0 else 0

        updated_content = f'Average Difference: {average_difference} Average Percentage Difference: {round(avg_difference_percentage, 2)}% Accurate predictions (+/- {accuracy_param}%): {accuracy}/{counter}\n'

        write_to_text_file(eval_dir, eval_file_name, updated_content)
    else:
        print(f'File: {data_file} does not exist')
        return
        
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

def create_histogram():
    return

def usage():
    print("Proper usage is: python model-eval.py [model_name] [directory_name_to_save_files] [path_to_eval_images] [accuracy_param]")
    print("[model_name] - name of the model you want to use to evaluate your data")
    print("[directory_name_to_save_files] - name of the directory in the current directory which will be created and .txt file with results will be saved into")
    print("[path_to_eval_images] - absolute path to the directory containing images to be evaluated")
    print("[accuracy_param] defines the maximum percentage difference of predicted and labelled age to be recognized as accurate prediction")

if __name__ == "__main__":
    __main__()
