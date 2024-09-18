from flask import Flask, redirect, render_template, request, session, jsonify, url_for
from flask_pymongo import PyMongo
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from flask_socketio import SocketIO, emit
import face_recognition
import cv2
import numpy as np
import pandas as pd
import os
from bson.objectid import ObjectId
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "Atharv"

# MongoDB Setup
app.config["MONGO_URI"] = (
    "mongodb+srv://atharva29:sZZQ4Xs0leba85zr@cluster0.hwxyvxb.mongodb.net/attendance?retryWrites=true&w=majority&appName=Cluster0"
)
mongo = PyMongo(app)

# Flask-Login Setup
login_manager = LoginManager()
login_manager.init_app(app)

# Flask-SocketIO Setup
socketio = SocketIO(app)


# User class for login
class User(UserMixin):
    def __init__(self, username, user_id):
        self.id = user_id
        self.username = username


@login_manager.user_loader
def load_user(user_id):
    user = mongo.db.users.find_one({"_id": ObjectId(user_id)})
    if user:
        return User(username=user["username"], user_id=str(user["_id"]))
    return None


# Login Route
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = mongo.db.users.find_one({"username": username})

        if user and check_password_hash(user["password"], password):
            login_user(User(username=user["username"], user_id=str(user["_id"])))
            return redirect(url_for("dashboard"))
        return "Invalid credentials"
    return render_template("login.html")


# Registration Route
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = generate_password_hash(request.form["password"])
        role = request.form["role"]  # "student" or "professor"

        mongo.db.users.insert_one(
            {"username": username, "password": password, "role": role, "encodings": []}
        )
        return redirect(url_for("login"))
    return render_template("register.html")


# Dashboard
@app.route("/dashboard")
@login_required
def dashboard():
    user = mongo.db.users.find_one({"_id": ObjectId(session["_user_id"])})
    return render_template("dashboard.html", user=user)


# Upload photo for face encoding
@app.route("/upload_photo", methods=["POST"])
@login_required
def upload_photo():
    user = mongo.db.users.find_one({"_id": ObjectId(session["_user_id"])})

    if user["role"] != "student":
        return "Unauthorized Access"

    if "photo" in request.files:
        photo = request.files["photo"]
        file_path = os.path.join("static/photos", f"{user['username']}.jpg")
        photo.save(file_path)

        image = face_recognition.load_image_file(file_path)
        face_encoding = face_recognition.face_encodings(image)

        if face_encoding:
            mongo.db.users.update_one(
                {"_id": ObjectId(session["_user_id"])},
                {"$set": {"encodings": [face_encoding[0].tolist()]}},
            )
            return "Photo Uploaded and Face Encoded"
        else:
            return "No face detected in the uploaded photo"

    return "No Photo Found"


# Capture photo from webcam for face encoding
@app.route("/capture_photo", methods=["POST"])
@login_required
def capture_photo():
    user = mongo.db.users.find_one({"_id": ObjectId(session["_user_id"])})

    if user["role"] != "student":
        return "Unauthorized Access"

    video_capture = cv2.VideoCapture(0)
    ret, frame = video_capture.read()

    if not ret:
        return "Failed to capture photo from webcam"

    video_capture.release()

    file_path = os.path.join("static/photos", f"{user['username']}.jpg")
    cv2.imwrite(file_path, frame)

    image = face_recognition.load_image_file(file_path)
    face_encoding = face_recognition.face_encodings(image)

    if face_encoding:
        mongo.db.users.update_one(
            {"_id": ObjectId(session["_user_id"])},
            {"$set": {"encodings": [face_encoding[0].tolist()]}},
        )
        return "Photo Captured from Webcam and Face Encoded"
    else:
        return "No face detected in the webcam capture"


@socketio.on("start_attendance_session")
def start_attendance_session(data):
    global attendance_data
    session_name = data["session_name"]
    date = data["date"]
    time = data["time"]
    attendance_data = {
        "session_name": session_name,
        "date": date,
        "time": time,
        "students": set(),
    }
    emit("status", {"message": "Attendance session started"})


@socketio.on("stop_attendance_session")
def stop_attendance_session():
    global attendance_data
    if attendance_data:
        attendance_data["students"] = list(attendance_data["students"])
        mongo.db.attendance_sessions.insert_one(attendance_data)
        # Save attendance to Excel
        df = pd.DataFrame(list(attendance_data["students"]), columns=["Student Name"])
        file_name = f"attendance/attendance_{attendance_data['time']}_{attendance_data['session_name']}_{attendance_data['date']}.xlsx"

        # Save the DataFrame to Excel
        df.to_excel(file_name, index=False)
        attendance_data = None
    emit("status", {"message": "Attendance session stopped"})


@socketio.on("frame")
def handle_frame(arrayBuffer):
    global attendance_data
    if attendance_data:
        # Convert ArrayBuffer to numpy array
        np_array = np.frombuffer(arrayBuffer, dtype=np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        rgb_frame = image[:, :, ::-1]

        # Load known face encodings
        known_face_encodings = []
        known_face_names = []

        students = mongo.db.users.find({"role": "student"})
        for student in students:
            encodings = student.get("encodings", [])
            if encodings:
                known_face_encodings.append(np.array(encodings[0]))
                known_face_names.append(student["username"])

        # Detect faces
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        detected_faces = set()

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding
            )
            face_distances = face_recognition.face_distance(
                known_face_encodings, face_encoding
            )
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                detected_faces.add(name)

        if attendance_data:
            attendance_data["students"].update(detected_faces)

        emit("face_data", {"faces": list(detected_faces)})


# HTML Routes
@app.route("/take_attendance")
@login_required
def take_attendance():
    return render_template("take_attendance.html")


# CRUD Endpoints for Attendance Sessions


# View all attendance sessions
@app.route("/attendance_sessions")
@login_required
def view_attendance_sessions():
    sessions = mongo.db.attendance_sessions.find()
    return render_template("attendance_sessions.html", sessions=sessions)


# Edit attendance session
@app.route("/edit_attendance/<session_id>", methods=["GET", "POST"])
@login_required
def edit_attendance(session_id):
    session_data = mongo.db.attendance_sessions.find_one({"_id": ObjectId(session_id)})

    if request.method == "POST":
        session_name = request.form["session_name"]
        attendance_date = request.form["date"]
        attendance_time = request.form["time"]
        students = request.form.getlist("students")

        mongo.db.attendance_sessions.update_one(
            {"_id": ObjectId(session_id)},
            {
                "$set": {
                    "session_name": session_name,
                    "date": attendance_date,
                    "time": attendance_time,
                    "students": students,
                }
            },
        )
        return redirect(url_for("view_attendance_sessions"))

    return render_template("edit_attendance.html", session=session_data)


# Delete attendance session
@app.route("/delete_attendance/<session_id>")
@login_required
def delete_attendance(session_id):
    mongo.db.attendance_sessions.delete_one({"_id": ObjectId(session_id)})
    return redirect(url_for("view_attendance_sessions"))


# Logout Route
@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


if __name__ == "__main__":
    socketio.run(app, debug=True)
