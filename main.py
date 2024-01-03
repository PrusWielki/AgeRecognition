import tkinter as tk
import tkinter.ttk as ttk
import tkinter.font as fnt
from tkinter import filedialog
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import os
import shutil
import copy
from age_detection import AgeDetectionClass
import argparse



customTracker = False
root = tk.Tk()
videoPlayer = tk.Label(root)
face_cascade = cv2.CascadeClassifier('FaceRecognition/haarsascade_frontalface_default.xml')
facetracker = load_model('FaceRecognition\\facetracker.keras')
target_directory = "face_photos"
selected_option = None
previous_age = np.zeros(100)
detector = cv2.FaceDetectorYN()
yuNetModel = cv2.FaceDetectorYN.create(
            model='face_detection_yunet_2023mar.onnx',
            config="",
            input_size=[320, 320],
            score_threshold=0.9,
            nms_threshold=0.3,
            top_k=5000,
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU)

def str2bool(v):
    if v.lower() in ['on', 'yes', 'true', 'y', 't']:
        return True
    elif v.lower() in ['off', 'no', 'false', 'n', 'f']:
        return False
    else:
        raise NotImplementedError


def browse_video():
    file_path = filedialog.askopenfilename(filetypes=[("MP4 Files", "*.mp4")])

    return file_path

def browse_photos():
    file_paths = filedialog.askopenfilenames(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])

    return file_paths

def resize_frame(frame):
    new_size = min(len(frame), len(frame[0]))
    frame = cv2.resize(frame, (new_size, new_size))
    return frame

def detect_faces_haarcascade(image):
    height, width, _ = image.shape

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load the pre-trained Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
    
    # Extract coordinates of each face
    face_coordinates = []
    for (x, y, w, h) in faces:
        face_coordinates.append([x / width, y / height, (x + w) / width, (y + h) / height])
    
    return face_coordinates

def detect_faces_yunet(image):
    h, w, _ = image.shape
    yuNetModel.setInputSize([w, h])
    faces = yuNetModel.detect(image)
    face_coordinates = []
    if faces[1] is not None:
        for det in faces[1]:
            bbox = det[0:4].astype(np.int32)
            face_coordinates.append([bbox[0]/w,bbox[1]/h,(bbox[0]+bbox[2])/w,(bbox[1]+bbox[3])/h])
    return face_coordinates

def save_images(image_array):
    # Create the target directory if it doesn't exist
    try:
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)
        else:
            # If the directory exists, delete all existing photos
            file_list = os.listdir(target_directory)
            for file_name in file_list:
                file_path = os.path.join(target_directory, file_name)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)

        # Save the new images in the target directory
        for i, image in enumerate(image_array):
            Image.fromarray(image.astype(np.uint8)).save(os.path.join(target_directory, f"image_{i}.png"))
    except:
        print("there was an error saving image")


def cut_photo(image, size, coordinates):
    left, top, right, bottom = coordinates

    height, width = size, size
    left_pixel = int(left * width)
    top_pixel = int(top * height)
    right_pixel = int(right * width)
    bottom_pixel = int(bottom * height)
    
    cropped_image = image[top_pixel:bottom_pixel, left_pixel:right_pixel, :]

    return cropped_image

def show_rectangle(frame, sample_coords, size):
    # Controls the main rectangle
    cv2.rectangle(frame,
                  tuple(np.multiply(sample_coords[:2], [size, size]).astype(int)),
                  tuple(np.multiply(sample_coords[2:], [size, size]).astype(int)),
                  (255, 0, 0), 2)
    # Controls the label rectangle
    cv2.rectangle(frame,
                  tuple(np.add(np.multiply(sample_coords[:2], [size, size]).astype(int),
                               [0, -30])),
                  tuple(np.add(np.multiply(sample_coords[:2], [size, size]).astype(int),
                               [50, 0])),
                  (255, 0, 0), -1)

def calculate_age(sample_coords, size, i):
    try:
        img_path = os.path.join(target_directory, f"image_{i}.png")
        age = AgeDetectionClass.predict_age(img_path, sample_coords, size)

        global previous_age
        previous_age[i] = age
    except:
        print("there was no image")
        return 0

    return age

def show_age(frame, sample_coords, size, i, age):
    # Controls the text rendered
    cv2.putText(frame, f'{age}', tuple(np.add(np.multiply(sample_coords[:2], [size, size]).astype(int),
                                            [0, -5])),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


def update_view(frame, calculateAge = True):
    frame = resize_frame(frame)
    size = len(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    start_frame = copy.deepcopy(rgb)
    resized = tf.image.resize(rgb, (120, 120))

    showFrame = True
    images = []
    faces = []

    current_option = selected_option.get()

    if current_option == 1:
        faces = detect_faces_haarcascade(frame)
    elif current_option == 2:
        yhat = facetracker.predict(np.expand_dims(resized / 255, 0))
        faces.append(yhat[1][0])
        showFrame = yhat[0] > 0.5
    elif current_option == 3:
        faces = detect_faces_yunet(frame)

    global previous_age

    if showFrame:
        for sample_coords in faces:
            show_rectangle(frame, sample_coords, size)
            images.append(cut_photo(start_frame, size, sample_coords))

        save_images(images)

        if calculateAge:
            previous_age = list(range(100))

        for i, sample_coords in enumerate(faces):
            age = calculate_age(sample_coords, size, i)
            show_age(frame, sample_coords, size, i, age)

    cv2.imshow('EyeTrack', frame)

def show_video(cap):
    cv2.namedWindow('EyeTrack', cv2.WINDOW_NORMAL)
    i = 7
    calculateAge = True
    while cap.isOpened():
        _, frame = cap.read()
        update_view(frame, calculateAge)

        cv2.setWindowProperty('EyeTrack', cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_NORMAL)

        if (i > 21):
            calculateAge = True
            i = 0
        else:
            calculateAge = False

        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('EyeTrack', cv2.WND_PROP_VISIBLE) < 1:
            break

        i += 1
            
    cap.release()
    cv2.destroyAllWindows()

def show_photo(image_paths):
    cv2.namedWindow('EyeTrack', cv2.WINDOW_NORMAL)
    for image_path in image_paths:
        frame = cv2.imread(image_path)
        update_view(frame)

        cv2.setWindowProperty('EyeTrack', cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_NORMAL)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

def AddVideoEvent():
    path = browse_video()
    cap = cv2.VideoCapture(path)
    show_video(cap)


def AddCameraEvent():
    cap = cv2.VideoCapture(0)
    show_video(cap)

def AddPhotosEvent():
    path = browse_photos()
    show_photo(path)

# def ChangeTrackerState():
#     global customTracker
#     if customTracker:
#         button4.configure(text="Custom tracker off")
#         customTracker = False
#     else:
#         button4.configure(text="Custom tracker on")
#         customTracker = True

def setup_main():
    customTracker = False
    # Set the window title
    root.title("Age Detection")

    # Get the screen width and height
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Calculate the x and y positions for the centered window
    window_width = 900
    window_height = 600
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2

    # Set the window size and position
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")

    # Create a frame to hold the buttons
    button_frame = ttk.Frame(root)
    button_frame.pack()

    # Create and pack four buttons in the frame
    button1 = ttk.Button(button_frame, text="Add Video", command=AddVideoEvent)
    button1.grid(row=0, column=0, padx=10, pady=10)

    button2 = ttk.Button(button_frame, text="Use camera", command=AddCameraEvent)
    button2.grid(row=0, column=1, padx=10, pady=10)

    button3 = ttk.Button(button_frame, text="Add Photos", command=AddPhotosEvent)
    button3.grid(row=0, column=2, padx=10, pady=10)

    # global button4
    # button4 = ttk.Button(button_frame, text="Custom tracker off", command=ChangeTrackerState)
    # button4.grid(row=0, column=3, padx=10, pady=10)

    header = tk.Label(button_frame, text="Pick face detection model", font=("Arial", 12))
    header.grid(row=1, column=1)

    global selected_option
    selected_option = tk.IntVar()
    selected_option.set(1)

    # Create the radio button
    radio1 = tk.Radiobutton(button_frame, text="Haarcascade", variable=selected_option, value=1)
    radio1.grid(row=2, column=1, padx=10, pady=10)

    radio2 = tk.Radiobutton(button_frame, text="YuNet", variable=selected_option, value=3)
    radio2.grid(row=3, column=1, padx=10, pady=10)

    radio3 = tk.Radiobutton(button_frame, text="Our own model", variable=selected_option, value=2)
    radio3.grid(row=4, column=1, padx=10, pady=10)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Create the main application window

    # init_model_yunet()

    setup_main()
    # Start the Tkinter main loop
    root.mainloop()
