# import cv2
from gtts import gTTS
import pyttsx3

# Initialize the text-to-speech engine
engine = pyttsx3.init()


# Function to convert text to speech
def text_to_speech(text, lang='en'):
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save("temp.mp3")
    engine.stop()  # Stop any ongoing speech
    engine.say(text)
    engine.runAndWait()

text_to_speech("Hello Yaa")


# Your traffic sign detection logic here
# def detect_traffic_sign(frame):
#     # Replace this with your traffic sign detection logic
#     # ...

#     # Assuming you detected a traffic sign
#     detected_sign_text = "Stop sign ahead"
#     text_to_speech(detected_sign_text)

# # Example usage
# if __name__ == "__main__":
#     # Capture video from your camera or use a video file
#     cap = cv2.VideoCapture(0)  # Use 0 for default camera

#     # while True:
#     #     ret, frame = cap.read()

#     #     # Your traffic sign detection logic
#     #     # detect_traffic_sign(frame)

#     #     # Display the frame (you might want to remove this in a real application)
#     #     # cv2.imshow("Traffic Sign Detection", frame)

#     #     if cv2.waitKey(1) & 0xFF == ord('q'):
#     #         break

#     cap.release()
#     cv2.destroyAllWindows()
