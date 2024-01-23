import argparse
import cv2 as cv
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import copy
import numpy as np

facetracker = load_model('FaceRecognition\\facetracker.keras')
def resize_frame(frame):
    new_size = min(len(frame), len(frame[0]))
    frame = cv.resize(frame, (new_size, new_size))
    return frame

DIR = 'C:/Users/prusak.patryk/Documents/studia/ItML/Project/UTKFace'

if __name__ == '__main__':
    model = cv.FaceDetectorYN.create(
            model='face_detection_yunet_2023mar.onnx',
            config="",
            input_size=[320, 320],
            score_threshold=0.9,
            nms_threshold=0.3,
            top_k=5000,
            backend_id=cv.dnn.DNN_BACKEND_OPENCV,
            target_id=cv.dnn.DNN_TARGET_CPU)

    if DIR is not None:
        totalAmountOfFiles = len([name for name in os.listdir(DIR) if os.path.isfile(name)])
        noOfFacesDetected = 0
        for entry in os.scandir(DIR):
            filename = os.fsdecode(entry.path)
            if filename.endswith(".jpg"): 
                image = cv.imread(filename)
                h, w, _ = image.shape
                model.setInputSize([w, h])
                faces = model.detect(image)
                if faces[1] is not None:
                    noOfFacesDetected = noOfFacesDetected+1
                continue
            else:
                continue

        print("Amount of total faces detected: (YuNet) " + str(noOfFacesDetected))
        accuracy=noOfFacesDetected/totalAmountOfFiles*100
        print("Accuracy: (YuNet) " + str(accuracy)+"%")
        noOfFacesDetected = 0
        for entry in os.scandir(DIR):
            filename = os.fsdecode(entry.path)
            if filename.endswith(".jpg"): 
                image = cv.imread(filename)
                height, width, _ = image.shape

                # Convert the image to grayscale
                gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
                # Load the pre-trained Haar Cascade classifier for face detection
                face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
                # Detect faces in the image
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=5, minSize=(20, 20))
    
                if len(faces)>0:
                    noOfFacesDetected = noOfFacesDetected+1
                continue
            else:
                continue

        print("Amount of total faces detected: (HaarCascade)" + str(noOfFacesDetected))
        accuracy=noOfFacesDetected/totalAmountOfFiles*100
        print("Accuracy: (HaarCascade) " + str(accuracy)+"%")
        noOfFacesDetected = 0
        for entry in os.scandir(DIR):
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

        print("Amount of total faces detected: (Own Model)" + str(noOfFacesDetected))
        accuracy=noOfFacesDetected/totalAmountOfFiles*100
        print("Accuracy: (Own Model) " + str(accuracy)+"%")
