import argparse
import cv2 as cv
import os

parser = argparse.ArgumentParser(description='YuNet: A Fast and Accurate CNN-based Face Detector (https://github.com/ShiqiYu/libfacedetection).')
parser.add_argument('--input', '-i', type=str,
                    help='Usage: Set input to a directory')
args = parser.parse_args()

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

    if args.input is not None:
        noOfFacesDetected = 0
        for entry in os.scandir(args.input):
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

        print("Amount of total faces detected: " + str(noOfFacesDetected))