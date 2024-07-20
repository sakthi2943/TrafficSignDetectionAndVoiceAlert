
# Traffic Sign Board Detection and Voice Alert System

This project is a Traffic Sign Board Detection and Voice Alert System developed as my university final project. The system detects traffic signs in real-time and provides voice alerts to the user. It is built using Python, TensorFlow, gTTS (Google Text-to-Speech), and OpenCV.

## Features

- **Real-time Traffic Sign Detection**: Utilizes OpenCV for live video capture and processing.
- **Voice Alerts**: Uses gTTS to convert detected traffic sign information into voice alerts.
- **Machine Learning**: Employs TensorFlow for training and deploying the traffic sign detection model.

## Technologies Used

- **Python**: Main programming language for the project.
- **TensorFlow**: Used for building and training the traffic sign detection model.
- **gTTS**: Google Text-to-Speech API for generating voice alerts.
- **OpenCV**: For real-time video capture and image processing.

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/traffic-sign-detection.git
   cd traffic-sign-detection
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Traffic Sign Detection and Voice Alert System**
   ```bash
   python detection.py
   ```

2. **The system will start capturing live video and detect traffic signs in real-time. Voice alerts will be provided for detected signs.**

## Acknowledgements

- [TensorFlow](https://www.tensorflow.org/)
- [gTTS](https://gtts.readthedocs.io/en/latest/)
- [OpenCV](https://opencv.org/)
- [German Traffic Sign Recognition Benchmark (GTSRB)](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)
