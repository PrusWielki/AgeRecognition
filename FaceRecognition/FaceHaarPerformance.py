import argparse
import cv2 as cv
import os

parser = argparse.ArgumentParser(description='Haar Cascade face recognition')
parser.add_argument('--input', '-i', type=str,
                    help='Usage: Set input to a directory')
args = parser.parse_args()

if __name__ == '__main__':

    if args.input is not None:
        noOfFacesDetected = 0
        for entry in os.scandir(args.input):
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

        print("Amount of total faces detected: " + str(noOfFacesDetected))