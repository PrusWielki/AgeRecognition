import tkinter as tk
import tkinter.ttk as ttk
import tkinter.font as fnt
from tkinter import filedialog
from tkvideo import tkvideo
import cv2

root = tk.Tk()
videoPlayer = tk.Label(root)

def browse_video():
    file_path = filedialog.askopenfilename(filetypes=[("MP4 Files", "*.mp4")])

    return file_path

def AddVideoEvent():
    path = browse_video()
    play_video(path)

def play_video(file_path):
    videoPlayer.pack()
    player = tkvideo(file_path, videoPlayer, loop=1, size=(500, 500))
    player.play()


def AddCameraEvent():
    pass

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
