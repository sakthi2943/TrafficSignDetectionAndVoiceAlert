from keras.models import load_model
import cv2
import numpy as np
import threading

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("../Model/model.savedmodel", compile=False)

# Load the labels
class_names = open("../Model/labels.txt", "r").readlines()

# Set the desired frame dimensions
frame_width = 1280  # Adjust to your desired width
frame_height = 720  # Adjust to your desired height

# CAMERA can be 0 or 1 based on the default camera of your computer
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Function to perform prediction
def predict_frame(model, frame):
    # Resize the raw image into (224-height,224-width) pixels
    resized_frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)

    # Make the image a numpy array and reshape it to the model's input shape.
    image = np.asarray(resized_frame, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Overlay prediction on the image
    cv2.putText(frame, f"Class: {class_name[2:]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Confidence: {str(np.round(confidence_score * 100))[:-2]}%", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the image in a window
    cv2.imshow("Webcam Image", frame)

# Function to capture frames from the webcam
def webcam_capture():
    while True:
        ret, image = camera.read()
        if not ret:
            break

        # Perform prediction in a separate thread
        prediction_thread = threading.Thread(target=predict_frame, args=(model, image))
        prediction_thread.start()

        # Wait for the prediction thread to finish before moving to the next frame
        prediction_thread.join()

        # Listen to the keyboard for presses.
        keyboard_input = cv2.waitKey(1)

        # 's' key to save the current frame
        if keyboard_input == ord('s'):
            cv2.imwrite("captured_frame.jpg", image)
        # 'Esc' key to exit
        elif keyboard_input == 27:
            break

# Start webcam capture in a separate thread
capture_thread = threading.Thread(target=webcam_capture)
capture_thread.start()

# Release the camera and close all OpenCV windows when done
capture_thread.join()
camera.release()
cv2.destroyAllWindows()
