from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img
import numpy as np
from PIL import Image
import cv2

agerecognition = load_model('AgeRecognitionModel.keras')


class AgeDetectionClass:
    @staticmethod
    def get_image_features(image):
        img = load_img(image, grayscale=True)
        img = img.resize((128, 128), Image.LANCZOS)
        img = np.array(img)
        img = img.reshape(1, 128, 128, 1)
        img = img / 255.0
        return img

    # @staticmethod
    # def extract_roi(image, x, y, w, h):
    #     return image[y:y + h, x:x + w]

    @staticmethod
    def extract_roi(image, coords, size):
        start = tuple(np.add(np.multiply(coords[:2], [size, size]).astype(int), [0, -30]))
        end = tuple(np.add(np.multiply(coords[:2], [size, size]).astype(int), [80, 0]))

        return image[start[1]:end[1], start[0]:end[0]]
    @staticmethod
    def predict_age(img_to_test, coords, size):
        # new_image = AgeDetectionClass.extract_roi(img_to_test, coords, size)
        features = AgeDetectionClass.get_image_features(img_to_test)
        pred = agerecognition.predict(features)
        age = round(pred[0][0])
        return age
