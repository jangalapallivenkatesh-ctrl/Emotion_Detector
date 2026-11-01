# Emotion Detector

An AI-powered facial emotion detection web application built using Python, Flask, OpenCV, and Deep Learning. The application detects human facial expressions and predicts emotions from facial images or camera input, depending on the implemented application features.

## Features

- Facial detection using OpenCV Haar Cascade classifiers.
- Facial emotion classification using a trained Mini-Xception deep learning model.
- Python-based backend using Flask.
- Web interface using HTML, CSS, and JavaScript.
- User data management using SQLite.
- Organized model, templates, and static files.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Backend programming |
| Flask | Web application framework |
| OpenCV | Face detection and image processing |
| Mini-Xception | Emotion classification model |
| TensorFlow / Keras | Deep learning model support |
| HTML, CSS, JavaScript | Frontend development |
| SQLite | Database management |

## Project Structure

```text
Emotion_Detector/
│
├── app.py
├── users.db
├── .gitignore
├── README.md
│
├── haarcascade_files/
│   └── Haar Cascade XML files
│
├── models/
│   └── cnn.py
│
├── static/
│   └── images/
│
└── templates/
    └── HTML templates
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/jangalapallivenkatesh-ctrl/Emotion_Detector.git
```

### 2. Navigate to the Project Folder

```bash
cd Emotion_Detector
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

Install the packages required by your application:

```bash
pip install flask opencv-python numpy tensorflow
```

If your project uses additional packages, install them as well.

### 6. Run the Application

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000/
```

## How It Works

1. The application receives facial image or camera input.
2. OpenCV detects the face using Haar Cascade classifiers.
3. The detected facial region is processed for emotion prediction.
4. The trained Mini-Xception model classifies the facial expression.
5. The application displays the predicted emotion through its web interface.

## Deep Learning Model

This project uses a Mini-Xception-based model for facial emotion classification.

The model file included in the project is:

```text
_mini_XCEPTION.102-0.66.hdf5
```

## Future Enhancements

- Improve emotion prediction accuracy.
- Add real-time emotion detection.
- Support multiple face detection.
- Improve the user interface.
- Add emotion analysis reports and visualizations.

## Author

**jangalapallivenkatesh-ctrl**

GitHub: https://github.com/jangalapallivenkatesh-ctrl

## License

This project is available for educational and learning purposes.
