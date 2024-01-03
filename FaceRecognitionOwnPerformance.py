import argparse
import cv2 as cv
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import copy

parser = argparse.ArgumentParser(description='YuNet: A Fast and Accurate CNN-based Face Detector (https://github.com/ShiqiYu/libfacedetection).')
parser.add_argument('--input', '-i', type=str,
                    help='Usage: Set input to a directory')
args = parser.parse_args()

facetracker = load_model('FaceRecognition\\facetracker.keras')

def resize_frame(frame):
    new_size = min(len(frame), len(frame[0]))
    frame = cv.resize(frame, (new_size, new_size))
    return frame

if __name__ == '__main__':

    if args.input is not None:
        noOfFacesDetected = 0
        for entry in os.scandir(args.input):
            filename = os.fsdecode(entry.path)
            if filename.endswith(".jpg"): 
                faces = []
                image = cv.imread(filename)
                frame = resize_frame(image)
                size = len(frame)

                rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                start_frame = copy.deepcopy(rgb)
                resized = tf.image.resize(rgb, (120, 120))
                yhat = facetracker.predict(np.expand_dims(resized / 255, 0))
                faces.append(yhat[1][0])
                if yhat[0] > 0.5:
                    noOfFacesDetected = noOfFacesDetected+1
                continue
            else:
                continue

        print("Amount of total faces detected: " + str(noOfFacesDetected))