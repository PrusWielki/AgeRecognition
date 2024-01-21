import sys
import os
import shutil
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import numpy as np
import time

current_directory = os.getcwd()
accuracy_in_percentage = False

age_group_size = 10

def __main__():
    if len(sys.argv) != 5:
        print(f"length: {len(sys.argv)}, argv: {sys.argv}")
        usage()
        return
    
    if len(sys.argv) == 6:
        evaluate_on_precentage = sys.argv[5].lower()
        evaluate_based_on_percentage = True if evaluate_on_precentage == 'true' else False
    
    # Load the pre-trained model
    model_name = sys.argv[1]
    model_path = os.path.join(current_directory, model_name)
    model = load_model(model_path)

    # Create directory or remove its content to save data
    evaluation_directory_name = sys.argv[2]
    evaluation_directory_path = os.path.join(current_directory, evaluation_directory_name)
    create_directory(evaluation_directory_path)
    eval_file_name = 'evaluation.txt'

    images_path = sys.argv[3]
    accuracy_param = int(sys.argv[4])
    test_images(images_path, model, evaluation_directory_path, accuracy_param)
    #read_and_prepare_eval_results(directory_path, eval_file_name, 'result.txt', accuracy_param)
    analyze_and_visualize_data(evaluation_directory_path, eval_file_name)

def test_images(directory_path, model, eval_dir, accuracy_param):
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
            eval_dataset(image_file, directory_path, model, eval_dir, accuracy_param)
        except Exception as e:
            print(f"Error processing {image_file}. Reason: {e}")

# specific logic for data from: https://www.kaggle.com/datasets/rashikrahmanpritom/age-recognition-dataset
# or other images that contains labelled age in the format: AGE_x_y_z.jpg (age must be in the first index separated by '_' sign)
def eval_dataset(image_file, dir_path, model, eval_dir, accuracy_param):
    img = get_image_features(os.path.join(dir_path, image_file))

    predictions = model.predict(img)
    age = round(predictions[0][0])

    splitted_text = image_file.split('_')
    labelled_age = int(splitted_text[0])
    difference = abs(labelled_age - age);
    percent_difference = abs(int(difference)) / labelled_age * 100
    is_accurate = round(percent_difference) <= accuracy_param if accuracy_in_percentage else round(difference) <= accuracy_param

    eval_file_name = 'evaluation.txt'
    obj = {
        'Model': age,
        'Label': labelled_age,
        'Difference': difference,
        'PercentageDifference': round(percent_difference),
        'Accurate': is_accurate
    }
    write_to_text_file(eval_dir, eval_file_name, str(obj))
    #write_to_text_file(eval_dir, eval_file_name, f'| Model:{age:3} | Label:{labelled_age:3} | Difference:{difference} | PercentageDifference:{str(round(percent_difference)):3} | Accurate:{round(percent_difference) <= accuracy_param:1} |')

def write_to_text_file(dir_path, file_name, content):
    file_path = os.path.join(dir_path, file_name)

    if os.path.exists(file_path):
        with open(file_path, 'a') as file:
            file.write(content + '\n')
    else:
        with open(file_path, 'w') as file:
            file.write(content + '\n')

def analyze_and_visualize_data(eval_dir, file_path):
    data_file = os.path.join(eval_dir, file_path)
    if not os.path.exists(data_file):
        print(f'There is no such a file at the path: {data_file}')
        
    with open(data_file, 'r') as file:
        lines = file.readlines()

    data_str = ''.join(lines).replace('\n', ',')

    data_list = eval('[' + data_str + ']')

    df = pd.DataFrame(data_list)
    print(df)
    # Oblicz RMSE i MAPE dla wszystkich danych
    rmse_all = np.sqrt(mean_squared_error(df['Label'], df['Model']))
    mape_all = np.mean(np.abs((df['Model'] - df['Label']) / df['Label'])) * 100

    print("RMSE for all data:", rmse_all)
    print("MAPE for all data:", mape_all)

    # Określ zakresy grup wiekowych
    age_ranges = list(range(0, 111, age_group_size))
    age_labels = [f"{start}-{start+age_group_size-1}" for start in age_ranges[:-1]]

    # Przypisz etykiety do poszczególnych grup wiekowych
    df['AgeGroup'] = pd.cut(df['Label'], bins=age_ranges, labels=age_labels, right=False)

    # Oblicz RMSE i MAPE dla grup wiekowych
    rmse_groups = []
    mape_groups = []

    for age_group in age_labels:
        group = df[df['AgeGroup'] == age_group]

        # Skip empty groups
        if group.empty:
            rmse_groups.append(0)
            mape_groups.append(0)
            continue

        rmse_group = np.sqrt(mean_squared_error(group['Label'], group['Model']))
        mape_group = np.mean(np.abs((group['Model'] - group['Label']) / group['Label'])) * 100
        rmse_groups.append(rmse_group)
        mape_groups.append(mape_group)
        print(f"\nRMSE for age group {age_group}:", rmse_group)
        print(f"MAPE for age group {age_group}:", mape_group)

    # Sprawdź liczbę poprawnych predykcji
    correct_predictions = df[df['Accurate'] == True].shape[0]
    total_predictions = df.shape[0]
    accuracy = correct_predictions / total_predictions * 100

    print("\nAccuracy:", accuracy, "%")

    # Zapisz wyniki do pliku
    results_dict = {
        'RMSE_all': rmse_all,
        'MAPE_all': mape_all,
        'RMSE_groups': rmse_groups,
        'MAPE_groups': mape_groups,
        'Accuracy': accuracy
    }

    result_file_path = os.path.join(eval_dir, 'result.txt')
    with open(result_file_path, 'w') as results_file:
        results_file.write(str(results_dict))

    # Generuj wykresy
    labels = [str(age_group) for age_group in age_labels]

    plt.figure(figsize=(10, 6))

    # Wykres RMSE dla grup wiekowych
    plt.subplot(2, 2, 1)
    plt.bar(labels, rmse_groups, color='blue')
    plt.title('RMSE for Age Groups')
    plt.xlabel('Age Groups')
    plt.ylabel('RMSE')

    # Wykres MAPE dla grup wiekowych
    plt.subplot(2, 2, 2)
    plt.bar(labels, mape_groups, color='orange')
    plt.title('MAPE for Age Groups')
    plt.xlabel('Age Groups')
    plt.ylabel('MAPE')

    # Wykres dokładności predykcji
    plt.subplot(2, 2, 3)
    plt.pie([correct_predictions, total_predictions - correct_predictions], labels=['Correct', 'Incorrect'], autopct='%1.1f%%', colors=['green', 'red'])
    plt.title('Accuracy of Predictions')

    # Wykresy ogólne
    plt.tight_layout()
    plt.show()

def create_RMSE_plot(labels, rmse_groups, color):
    plt.figure(figsize=(10, 6))
    plt.plot()
    plt.bar(labels, rmse_groups, color)
    plt.title('RMSE for Age Groups')
    plt.xlabel('Age Groups')
    plt.ylabel('RMSE')
    plt.tight_layout()
    plt.show()

def create_MAPE_plot(labels, mape_groups, color):
    plt.figure(figsize=(10, 6))
    plt.plot()
    plt.bar(labels, mape_groups, color)
    plt.title('MAPE for Age Groups')
    plt.xlabel('Age Groups')
    plt.ylabel('MAPE')
    plt.tight_layout()
    plt.show()

def create_ACCURACY_plot(predictions, labels, autopct, colors):
    plt.figure(figsize=(10, 6))
    plt.plot()
    plt.pie(predictions, labels, autopct, colors)
    plt.title('Accuracy of Predictions')
    plt.tight_layout()
    plt.show()

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
    print("[accuracy_param] defines the maximum difference of predicted and labelled age to be recognized as accurate prediction")
    print("[evaluate_on_percentage] boolean value that defines whether the evaluation on accuracy of model should be based on percentage difference or not - default value is False")

if __name__ == "__main__":
    __main__()
