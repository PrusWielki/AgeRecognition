# 
# REMEMBER TO INSTALL PACKAGE
# pip install Pillow 
#
#
from PIL import Image
import os
import sys
import json
import random

def resize_images(input_face, input_no_face, output_path):
    try:
        new_image_folder_path = os.path.join(output_path, "images")
        new_label_folder_path = os.path.join(output_path, "labels")

        if not os.path.exists(new_image_folder_path):
            os.makedirs(new_image_folder_path)

        if not os.path.exists(new_label_folder_path):
            os.makedirs(new_label_folder_path)

        if not os.path.exists(input_face):
            print(f"Given path '{input_face}' does not exist.")
            return
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        labelsPath = input_face + "\\labels2"
        imagesPath = input_face + "\\images\\train"

        no_face_image_files = [file for file in input_no_face if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        
        noFaceImageCount = 0

        for filename in os.listdir(input_no_face):
            data_json = {
                    "image": filename,
                    "bbox": [
                        0, 0, 0, 0
                    ],
                    "class": 0
                }
            
            json_filename = os.path.join(new_label_folder_path, filename)
            json_filename = os.path.splitext(json_filename)[0] + ".json"

            with open(json_filename, 'w') as json_file:
                    json.dump(data_json, json_file)

            image_file_path = os.path.join(input_no_face, filename)
            
            with Image.open(image_file_path) as img:
                new_size = min(img.width, img.height)
                img = img.resize((new_size, new_size))
                output_file_path = os.path.join(new_image_folder_path, filename)
                img.save(output_file_path)

            noFaceImageCount += 1
            if (noFaceImageCount == 1500):
                break

        faceImageCount = 0

        for filename in os.listdir(labelsPath):
            input_file_path = os.path.join(labelsPath, filename)

            if not os.path.isfile(input_file_path):
                continue

            with open(input_file_path, 'r') as file:
                lines = file.readlines()
                if(len(lines) != 1):
                    continue
                data = lines[0].split()

            x = float(data[2])
            y = float(data[3])
            xEnd = float(data[4])
            yEnd = float(data[5])

            image_file_path = os.path.join(imagesPath, filename)
            image_file_path = os.path.splitext(image_file_path)[0] + ".jpg"

            if not os.path.isfile(image_file_path):
                continue


            with Image.open(image_file_path) as img:
                new_size = min(img.width, img.height)
                x = x / img.width * new_size / img.width
                y = y / img.height * new_size / img.height
                xEnd = xEnd / img.width * new_size / img.width
                yEnd = yEnd / img.height * new_size / img.height
                img = img.resize((new_size, new_size))
                output_file_path = os.path.join(new_image_folder_path, filename)
                output_file_path = os.path.splitext(output_file_path)[0] + ".jpg"
                img.save(output_file_path)

                data_json = {
                    "image": f"{os.path.splitext(filename)[0]}.jpg",
                    "bbox": [
                        x, y, xEnd, yEnd
                    ],
                    "class": 1
                }
                json_filename = os.path.join(new_label_folder_path, filename)
                json_filename = os.path.splitext(json_filename)[0] + ".json"
                with open(json_filename, 'w') as json_file:
                    json.dump(data_json, json_file)

            faceImageCount += 1
            if (faceImageCount == 1500):
                exit(0)

    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python resize_images.py [input_face] [input_no_face] [output_path]")
    else:
        input_face = sys.argv[1]
        input_no_face = sys.argv[2]
        output_path = sys.argv[3]
        resize_images(input_face, input_no_face, output_path)