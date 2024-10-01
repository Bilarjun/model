import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import cv2
from ultralytics import YOLO
from PIL import Image, ImageTk
import threading

# Load the YOLO weed detection model (replace with the actual path to your trained model)
model = YOLO("C:/Users/Arjun Bilupati/PycharmProjects/pythonProject6/runs/detect/train/weights/best.pt")  # Correct path to best.pt

# The class index for "weed" detection (update this with your actual class index for "weed")
WEED_CLASS_INDEX = 0  # Update this if necessary based on your dataset

# Initialize the camera
camera_index = 0  # Default to the first camera (changeable)
cap = None
running = False

# Start detection
def start_detection():
    global cap, running
    running = True
    cap = cv2.VideoCapture(camera_index)

    def detect():
        while running:
            ret, frame = cap.read()
            if ret:
                # Run YOLOv8 model on the frame
                results = model(frame)

                # Draw bounding boxes only for "weed" class
                for result in results:
                    for i, box in enumerate(result.boxes.xyxy):
                        cls = result.boxes.cls[i].item()  # Get class index
                        if cls == WEED_CLASS_INDEX:  # Filter only the "weed" class
                            x1, y1, x2, y2 = box.int().tolist()
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow box for weed

                # Convert OpenCV image to PIL format for Tkinter
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img = ImageTk.PhotoImage(image=img)

                # Update the label with the image
                camera_label.imgtk = img
                camera_label.configure(image=img)

        cap.release()

    threading.Thread(target=detect).start()

# Stop detection
def stop_detection():
    global running
    running = False

# Function to set camera index
def set_camera_index():
    global camera_index
    camera_index = int(camera_input.get())
    messagebox.showinfo("Camera Selection", f"Selected Camera: {camera_index}")

# GUI
root = tk.Tk()
root.title("Weed Detection Using YOLOv8")

# Camera feed display
camera_label = tk.Label(root)
camera_label.pack()

# Start and Stop buttons
start_button = tk.Button(root, text="Start Detection", command=start_detection)
start_button.pack(pady=10)

stop_button = tk.Button(root, text="Stop Detection", command=stop_detection)
stop_button.pack(pady=10)

# Camera selection
camera_input_label = tk.Label(root, text="Enter Camera Index:")
camera_input_label.pack()

camera_input = tk.Entry(root)
camera_input.pack()

camera_select_button = tk.Button(root, text="Select Camera", command=set_camera_index)
camera_select_button.pack(pady=10)

# Start the Tkinter main loop
root.mainloop()
