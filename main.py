import tkinter as tk
import tkinter.ttk as ttk
import tkinter.font as fnt
from tkinter import filedialog
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model

root = tk.Tk()
videoPlayer = tk.Label(root)
face_cascade = cv2.CascadeClassifier('FaceRecognition/haarsascade_frontalface_default.xml')
facetracker = load_model('FaceRecognition\\facetracker.keras')

def browse_video():
    file_path = filedialog.askopenfilename(filetypes=[("MP4 Files", "*.mp4")])

    return file_path

def show_video(cap):
    while cap.isOpened():
        _, frame = cap.read()
        frame = frame[50:500, 50:500, :]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = tf.image.resize(rgb, (120, 120))

        yhat = facetracker.predict(np.expand_dims(resized / 255, 0))
        sample_coords = yhat[1][0]

        if yhat[0] > 0.5:
            # Controls the main rectangle
            cv2.rectangle(frame,
                          tuple(np.multiply(sample_coords[:2], [450, 450]).astype(int)),
                          tuple(np.multiply(sample_coords[2:], [450, 450]).astype(int)),
                          (255, 0, 0), 2)
            # Controls the label rectangle
            cv2.rectangle(frame,
                          tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),
                                       [0, -30])),
                          tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),
                                       [80, 0])),
                          (255, 0, 0), -1)

            # Controls the text rendered
            cv2.putText(frame, 'face', tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),
                                                    [0, -5])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('EyeTrack', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def AddVideoEvent():
    path = browse_video()
    cap = cv2.VideoCapture(path)
    show_video(cap)


def AddCameraEvent():
    cap = cv2.VideoCapture(0)
    show_video(cap)

def AddPhotosEvent():
    pass

def setup_main():
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

    # Create and pack three buttons in the frame
    button1 = ttk.Button(button_frame, text="Add Video", command=AddVideoEvent)
    button1.grid(row=0, column=0, padx=10, pady=10)

    button2 = ttk.Button(button_frame, text="Use camera", command=AddCameraEvent)
    button2.grid(row=0, column=1, padx=10, pady=10)

    button3 = ttk.Button(button_frame, text="Add Photos", command=AddPhotosEvent)
    button3.grid(row=0, column=2, padx=10, pady=10)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Create the main application window

    setup_main()
    # Start the Tkinter main loop
    root.mainloop()
