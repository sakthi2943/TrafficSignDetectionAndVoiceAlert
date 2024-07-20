from tensorflow.keras.models import load_model  # type: ignore # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
from TrafficSignDetection import *

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("Model\model.savedmodel", compile=False)

# Load the labels
class_names = open("Model\labels.txt", "r").readlines()

# Set the desired frame dimensions
frame_width = 1280  # Adjust to your desired width
frame_height = 770  # Adjust to your desired height

# CAMERA can be 0 or 1 based on the default camera of your computer
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
while True:
    # Grab the webcamera's image.
    __, image = camera.read()

    # Contouring the detected sign image
    redness = returnRedness(image) 
    thresh = threshold(redness) 	
    try:
        contours = findContour(thresh)
        big = findBiggestContour(contours)
        if cv2.contourArea(big) > 3000:
            print(cv2.contourArea(big))
            img,sign = boundaryBox(image,big)
            cv2.imshow('Frame',img)
            cv2.imshow('Sign',sign)
        else:
            cv2.imshow('frame',image)
    except:
        cv2.imshow('frame',image)


    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    # cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    sign_detected = class_name[2:] + "Sign Detected"
    # print(sign_detected)


    if index != 0:
        text_to_speech(sign_detected)

    # Print prediction and confidence score
    print("Detected:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
