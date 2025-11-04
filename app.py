from flask import Flask,render_template,Response,request, send_from_directory,session, send_file, jsonify, request,flash,redirect,url_for
import sqlite3
import cv2
import os
import face_recognition
import numpy as np
import math
import datetime
import argparse
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, Colors
from copy import deepcopy
import pandas as pd

from datetime import datetime
sport_list = {
    'sit-up': {
        'left_points_idx': [6, 12, 14],
        'right_points_idx': [5, 11, 13],
        'maintaining': 70,
        'relaxing': 110,
        'concerned_key_points_idx': [5, 6, 11, 12, 13, 14],
        'concerned_skeletons_idx': [[14, 12], [15, 13], [6, 12], [7, 13]]
    },
    'pushup': {
        'left_points_idx': [6, 8, 10],
        'right_points_idx': [5, 7, 9],
        'maintaining': 140,
        'relaxing': 120,
        'concerned_key_points_idx': [5, 6, 7, 8, 9, 10],
        'concerned_skeletons_idx': [[9, 11], [7, 9], [6, 8], [8, 10]]
    },
    'squat': {
        'left_points_idx': [11, 13, 15],
        'right_points_idx': [12, 14, 16],
        'maintaining': 80,
        'relaxing': 140,
        'concerned_key_points_idx': [11, 12, 13, 14, 15],
        'concerned_skeletons_idx': [[16, 14], [14, 12], [17, 15], [15, 13]]
    }
}


def calculate_angle(key_points, left_points_idx, right_points_idx):
    def _calculate_angle(line1, line2):
        # Calculate the slope of two straight lines
        slope1 = math.atan2(line1[3] - line1[1], line1[2] - line1[0])
        slope2 = math.atan2(line2[3] - line2[1], line2[2] - line2[0])

        # Convert radians to angles
        angle1 = math.degrees(slope1)
        angle2 = math.degrees(slope2)

        # Calculate angle difference
        angle_diff = abs(angle1 - angle2)

        # Ensure the angle is between 0 and 180 degrees
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        return angle_diff

    left_points = [[key_points.data[0][i][0], key_points.data[0][i][1]] for i in left_points_idx]
    right_points = [[key_points.data[0][i][0], key_points.data[0][i][1]] for i in right_points_idx]
    line1_left = [
        left_points[1][0].item(), left_points[1][1].item(),
        left_points[0][0].item(), left_points[0][1].item()
    ]
    line2_left = [
        left_points[1][0].item(), left_points[1][1].item(),
        left_points[2][0].item(), left_points[2][1].item()
    ]
    angle_left = _calculate_angle(line1_left, line2_left)
    line1_right = [
        right_points[1][0].item(), right_points[1][1].item(),
        right_points[0][0].item(), right_points[0][1].item()
    ]
    line2_right = [
        right_points[1][0].item(), right_points[1][1].item(),
        right_points[2][0].item(), right_points[2][1].item()
    ]
    angle_right = _calculate_angle(line1_right, line2_right)
    angle = (angle_left + angle_right) / 2
    return angle


def plot(pose_result, plot_size_redio, show_points=None, show_skeleton=None):
    class _Annotator(Annotator):

        def kpts(self, kpts, shape=(640, 640), radius=5, line_thickness=2, kpt_line=True):
           
            if self.pil:
                # Convert to numpy first
                self.im = np.asarray(self.im).copy()
            nkpt, ndim = kpts.shape
            is_pose = nkpt == 17 and ndim == 3
            kpt_line &= is_pose  # `kpt_line=True` for now only supports human pose plotting
            colors = Colors()
            for i, k in enumerate(kpts):
                if show_points is not None:
                    if i not in show_points:
                        continue
                color_k = [int(x) for x in self.kpt_color[i]] if is_pose else colors(i)
                x_coord, y_coord = k[0], k[1]
                if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
                    if len(k) == 3:
                        conf = k[2]
                        if conf < 0.5:
                            continue
                    cv2.circle(self.im, (int(x_coord), int(y_coord)),
                               int(radius * plot_size_redio), color_k, -1, lineType=cv2.LINE_AA)

            if kpt_line:
                ndim = kpts.shape[-1]
                for i, sk in enumerate(self.skeleton):
                    if show_skeleton is not None:
                        if sk not in show_skeleton:
                            continue
                    pos1 = (int(kpts[(sk[0] - 1), 0]), int(kpts[(sk[0] - 1), 1]))
                    pos2 = (int(kpts[(sk[1] - 1), 0]), int(kpts[(sk[1] - 1), 1]))
                    if ndim == 3:
                        conf1 = kpts[(sk[0] - 1), 2]
                        conf2 = kpts[(sk[1] - 1), 2]
                        if conf1 < 0.5 or conf2 < 0.5:
                            continue
                    if pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
                        continue
                    if pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
                        continue
                    cv2.line(self.im, pos1, pos2, [int(x) for x in self.limb_color[i]],
                             thickness=int(line_thickness * plot_size_redio), lineType=cv2.LINE_AA)
            if self.pil:
                # Convert im back to PIL and update draw
                self.fromarray(self.im)

    annotator = _Annotator(deepcopy(pose_result.orig_img))
    if pose_result.keypoints is not None:
        for k in reversed(pose_result.keypoints.data):
            annotator.kpts(k, pose_result.orig_shape, kpt_line=True)
    return annotator.result()


def put_text(frame, exercise, count, fps, redio):
    cv2.rectangle(
        frame, (int(20 * redio), int(20 * redio)), (int(300 * redio), int(163 * redio)),
        (0, 93, 160), -1
    )

    if exercise in sport_list.keys():
        cv2.putText(
            frame, f'Exercise: {exercise}', (int(30 * redio), int(50 * redio)), 0, 0.9 * redio,
            (255, 255, 255), thickness=int(2 * redio), lineType=cv2.LINE_AA
        )
    elif exercise == 'No Object':
        cv2.putText(
            frame, f'No Object', (int(30 * redio), int(50 * redio)), 0, 0.9 * redio,
            (255, 255, 255), thickness=int(2 * redio), lineType=cv2.LINE_AA
        )
    cv2.putText(
        frame, f'Count: {count}', (int(30 * redio), int(100 * redio)), 0, 0.9 * redio,
        (255, 255, 255), thickness=int(2 * redio), lineType=cv2.LINE_AA
    )
    
app = Flask(__name__)
app.config['SECRET_KEY'] = '895623741'



database="6.db"

def createtable():
    conn=sqlite3.connect(database)
    cursor=conn.cursor()
    cursor.execute("create table if not exists register(id integer primary key autoincrement, name text,email text,password text,status text)")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            phone TEXT,
            address TEXT,
            image BLOB
        )
    """)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ExerciseData (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            exercise_type TEXT,
            count INTEGER,
            total_time TEXT,
            date_time TEXT
        )
    ''')
    conn.commit()
    conn.close()
createtable()

excel_file = "exercise_data.xlsx"


@app.route('/')
def home():
    return render_template('register.html')


@app.route('/register', methods=["GET","POST"])
def register():
    if request.method=="POST":
        name=request.form['name']
        email=request.form['email']

        password=request.form['password']
        conn=sqlite3.connect(database)
        cursor=conn.cursor()
        cursor.execute(" SELECT email FROM register WHERE email=?",(email,))
        registered=cursor.fetchall()
        if registered:
            return render_template('register.html', alert_message="Email Already Registered")
        else:
            cursor.execute("insert into register(name,email,password,status) values(?,?,?,?)",(name,email,password,0))
            conn.commit()
            return render_template('login.html', alert_message="Registered Succussfully")
    return render_template('register.html')



@app.route('/login', methods=["GET", "POST"])
def login():
    global data
    global email
    if request.method == "POST":        
        email = request.form['email']
        password = request.form['password']
        conn = sqlite3.connect(database)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM register WHERE email=? AND password=?", (email, password))
        data = cursor.fetchone()


        if data is None:
            return render_template('register.html', alert_message="Email Not Registered or Check Password")
        else:
            return render_template('dashboard.html')

    return render_template('login.html')

IMAGE_FOLDER = "face_images"
if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)


@app.route("/capture")
def capture():
    name = request.args.get("name")
    if not name:
        return jsonify({"error": "Name is required"}), 400

    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    captured = False

    while not captured:
        ret, img = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("frame", img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):  # Press 'c' to capture
            for (x, y, w, h) in faces:
                file_path = os.path.join(IMAGE_FOLDER, f"{name}.jpg")
                cv2.imwrite(file_path, gray[y:y+h, x:x+w])
                captured = True
                break

        elif key == ord("q"):  # Press 'q' to exit
            break

    cam.release()
    cv2.destroyAllWindows()

    if captured:
        return jsonify({"message": "Image Captured Successfully"}), 200
    else:
        return jsonify({"error": "Image capture failed"}), 500


@app.route("/dashboard")
def dashboard():
    return render_template('dashboard.html')

# Handle Form Submission
@app.route("/submit", methods=["POST"])
def submit():
    name = request.form["name"]
    email = request.form["email"]
    phone = request.form["phone"]
    address = request.form["address"]
    image_name = request.form["image_name"]

    image_path = os.path.join(IMAGE_FOLDER, image_name)
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            img_blob = img_file.read()
    else:
        return "Image not found!", 400

    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO users (name, email, phone, address, image) VALUES (?, ?, ?, ?, ?)",
                   (name, email, phone, address, img_blob))
    conn.commit()
    conn.close()

    return render_template('dashboard.html', alert_message="User Registered Successfully!")  

@app.route("/verification")
def verification():
    global name
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Error: Could not open video device.")
        return render_template('dashboard.html', alert_message="Error: Could not open camera."), 500

    path = 'face_images'
    files = os.listdir(path)

    face_images = [os.path.join(path, file) for file in files if file.endswith('.jpg') or file.endswith('.png')]
    print(f"Face images found: {face_images}")

    loaded_images = []
    encoded_images = []
    known_face_names = []

    for img in face_images:
        tmp = face_recognition.load_image_file(img)
        encodings = face_recognition.face_encodings(tmp)

        if len(encodings) > 0:  # Check if a face is detected
            enctmp = encodings[0]
            loaded_images.append(tmp)
            encoded_images.append(enctmp)
            known_face_names.append(os.path.splitext(os.path.basename(img))[0])
        else:
            print(f"Warning: No face detected in {img}")

    # If no faces were detected in any images, return an error message
    if not encoded_images:
        print("Error: No valid face encodings found. Please check the dataset.")
        video_capture.release()
        return render_template('dashboard.html', alert_message="No valid face encodings found. Please check the dataset."), 400

    known_face_encodings = encoded_images

    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Unable to capture video frame")
            break

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]  # Convert BGR to RGB

        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                if len(face_distances) > 0:  # Check if face_distances is not empty
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame

        # Draw rectangles around detected faces
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            print(name)

        cv2.imshow('Video', frame)

        # Press 'Esc' key to exit
        if cv2.waitKey(1) == 27:
            break

    video_capture.release()
    cv2.destroyAllWindows()

    if not face_names or "Unknown" in face_names:
        return render_template('dashboard.html', alert_message="Face not recognized. Please try again."), 400
    else:
        recognized_name = face_names[0] 
        return redirect(url_for('upload_video', name=recognized_name))


def save_to_db(name, exercise_type, count, total_time):
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO ExerciseData (name, exercise_type, count, total_time, date_time) VALUES (?, ?, ?, ?, ?)",
                   (name, exercise_type, count, total_time, timestamp))
    conn.commit()

    # Save to Excel
    save_to_excel(name, exercise_type, count, total_time, timestamp)

def save_to_excel(name, exercise_type, count, total_time, timestamp):
    """Save the data to an Excel file."""
    data = pd.DataFrame([[name, exercise_type, count, total_time, timestamp]],
                        columns=["Name", "Exercise Type", "Count", "Total Time (sec)", "Date & Time"])
    
    try:
        # Append to existing file
        with pd.ExcelWriter(excel_file, mode='a', if_sheet_exists='overlay', engine="openpyxl") as writer:
            data.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
    except FileNotFoundError:
        # Create a new file if not exists
        data.to_excel(excel_file, index=False)

PROCESSED_FOLDER = 'processed'
UPLOAD_FOLDER = 'uploads'

app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def fetch_exercise_data(name=None):
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    
    if name:
        query = "SELECT * FROM ExerciseData WHERE name LIKE ?"
        cursor.execute(query, (f"%{name}%",))
    else:
        query = "SELECT * FROM ExerciseData"
        cursor.execute(query)

    data = cursor.fetchall()
    conn.close()
    
    return [{"id": row[0], "name": row[1], "exercise_type": row[2], 
             "count": row[3], "total_time": row[4], "date_time": row[5]} for row in data]

# Route to show history page
@app.route("/history")
def history_page():
    name = request.args.get("name")  # Get search query if provided
    data = fetch_exercise_data(name)
    return render_template("history.html", records=data)


@app.route('/upload', methods=['GET', 'POST'])
def upload_video():
    model = 'yolo11n-pose.pt'

    if request.method == 'POST':
        if 'video' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['video']
        type_ste = request.form["type"]
        model = YOLO(model)

        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        processed_filepath = os.path.join(app.config['PROCESSED_FOLDER'], filename)
        process_video(filepath, processed_filepath, type_ste, model)

        return render_template('upload_video.html', alert_message="Verification Completed")
    
    return render_template('upload_video.html')

def process_video(input_path, output_path, type_ste, model):
    global name
    models = model
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    output = cv2.VideoWriter(os.path.join(output_path, 'result.mp4'), fourcc, fps, size)
    reaching = False
    reaching_last = False
    state_keep = False
    counter = 0
    start_time = datetime.now()


    while cap.isOpened():
        success, frame = cap.read()

        if success:
            plot_size_redio = max(frame.shape[1] / 960, frame.shape[0] / 540)

            results = models(frame)

            if results[0].keypoints.shape[1] == 0:
                    put_text(frame, 'No Object', counter,
                             round(1000 / results[0].speed['inference'], 2), plot_size_redio)
                    scale = 640 / max(frame.shape[0], frame.shape[1])
                    show_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                    cv2.imshow("YOLOv11 Inference", show_frame)
                    output.write(frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            left_points_idx = sport_list[type_ste]['left_points_idx']
            right_points_idx = sport_list[type_ste]['right_points_idx']

            angle = calculate_angle(results[0].keypoints, left_points_idx, right_points_idx)

            if angle < sport_list[type_ste]['maintaining']:
                reaching = True
            if angle > sport_list[type_ste]['relaxing']:
                reaching = False

            if reaching != reaching_last:
                reaching_last = reaching
                if reaching:
                    state_keep = True
                if not reaching and state_keep:
                    counter += 1
                    state_keep = False


            annotated_frame = plot(
                results[0], plot_size_redio,
            )

            put_text(
                annotated_frame, type_ste, counter, round(1000 / results[0].speed['inference'], 2), plot_size_redio)
            scale = 640 / max(annotated_frame.shape[0], annotated_frame.shape[1])
            show_frame = cv2.resize(annotated_frame, (0, 0), fx=scale, fy=scale)
            output.write(annotated_frame)
            cv2.imshow("YOLOv11 Inference", show_frame)
        else:
            break
    end_time = datetime.now()
    total_time = (end_time - start_time).seconds
    save_to_db(name, type_ste, counter, total_time)


    # Save final count and total time
    cap.release()
    output.release()

    cv2.destroyAllWindows()




if __name__ == "__main__":
    app.run(port=2000)
