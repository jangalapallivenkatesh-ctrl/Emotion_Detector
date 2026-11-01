from flask import Flask, render_template, Response, request, redirect, url_for, session, jsonify
import cv2
import imutils
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import sqlite3
import time
import os

app = Flask(__name__)
app.secret_key = "secretkey"


def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE,
                    phone TEXT,
                    password TEXT
                )""")
    conn.commit()
    conn.close()

init_db()


camera = None
streaming = False
detected_emotion_global = "Stopped"

detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

EMOTIONS = ["stress", "disgust", "scared", "happy", "sad", "surprised", "neutral"]


def generate_frames():
    global camera, streaming, detected_emotion_global

    while streaming and camera is not None:
        success, frame = camera.read()
        if not success:
            break

        frame = imutils.resize(frame, width=600)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detection.detectMultiScale(gray, 1.1, 5)

        for (fX, fY, fW, fH) in faces:
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = emotion_classifier.predict(roi)[0]
            detected_emotion = EMOTIONS[preds.argmax()]
            detected_emotion_global = detected_emotion

            cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (0, 255, 0), 2)
            cv2.putText(frame, detected_emotion, (fX, fY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            break

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')



@app.route('/')
def home():
    return render_template("home.html")

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == "POST":
        username = request.form['username']
        phone = request.form['phone']
        password = request.form['password']

        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, phone, password) VALUES (?, ?, ?)",
                      (username, phone, password))
            conn.commit()
        except:
            pass
        conn.close()
        return redirect(url_for('login'))

    return render_template("signup.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=? AND password=?",
                  (username, password))
        user = c.fetchone()
        conn.close()

        if user:
            session['user'] = username
            return redirect(url_for('dashboard'))

    return render_template("login.html")

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template("dashboard.html")


@app.route('/start')
def start():
    global camera, streaming, detected_emotion_global

    if camera is None:
        camera = cv2.VideoCapture(0)
        time.sleep(2)

    streaming = True
    detected_emotion_global = "Detecting..."
    return redirect(url_for('dashboard'))


@app.route('/stop')
def stop():
    global camera, streaming, detected_emotion_global

    streaming = False
    detected_emotion_global = "Stopped"

    if camera is not None:
        camera.release()
        camera = None

    return redirect(url_for('dashboard'))

@app.route('/video_feed')
def video_feed():
    if streaming:
        return Response(generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return ""


@app.route('/get_emotion')
def get_emotion():
    return jsonify({"emotion": detected_emotion_global})

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
