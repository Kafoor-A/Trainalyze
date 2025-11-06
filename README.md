# 🏋️‍♂️ Trainalyze – Smart Gym Activity Tracker

![Trainalyze Banner](https://github.com/Kafoor-A/Trainalyze/blob/main/1762258325980.jpg)

**Trainalyze** is an AI-powered gym analytics system that leverages **YOLO-based motion detection** and **ESP32 integration** to monitor gym members’ activities in real time.  
It identifies workout types, tracks performance, and provides progress insights for both users and trainers — enabling smarter, data-driven fitness monitoring.

---

## 🚀 Features
 - ✅ Real-time person detection and motion tracking  
 - ✅ Automatic exercise recognition using YOLOv8  
 - ✅ Individual workout performance analytics  
 - ✅ Trainer dashboard for client monitoring  
 - ✅ Cloud-based progress tracking and history  
 - ✅ ESP32 integration for sensor-based gym data collection  
 - ✅ User and Trainer portal for accessing reports  

---

## 🛠️ Tech Stack

| Layer | Technology |
|:------|:------------|
| **AI/ML Model** | YOLOv8 (Ultralytics) |
| **Programming Language** | Python |
| **Computer Vision** | OpenCV |
| **Backend** | Flask |
| **Database** | Firebase |
| **Microcontroller** | ESP32 |
| **IDE** | VS Code |
| **Version Control** | Git & GitHub |

---

## 🧩 Project Structure

```
Trainalyze/
│
├── main.py                     # Main entry point for the application
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
│
├── /models/                     # YOLO models and training files
│   ├── yolov8n.pt
│   └── custom_training.py
│
├── /dataset/                    # Training and testing datasets
│   ├── images/
│   └── labels/
│
├── /modules/                    # Core project modules
│   ├── detection.py             # YOLO detection logic
│   ├── tracking.py              # Person/workout tracking logic
│   ├── analytics.py             # Workout analytics computation
│   └── firebase_utils.py        # Firebase database connection and upload
│
├── /templates/                  # Flask HTML templates
│   ├── index.html               # Home dashboard
│   ├── trainer.html             # Trainer interface
│   └── user.html                # User view
│
├── /static/                     # Static assets (CSS, JS, images)
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── main.js
│   └── images/
│       └── logo.png
│
└── /esp32/                      # Hardware integration files
    ├── esp32_script.ino         # ESP32 data upload code
    └── sensors/                 # Sensor interfacing scripts
```


---

## ⚙️ Installation & Setup

### 🔹 Prerequisites
- Python 3.10+  
- Git  
- VS Code  
- YOLOv8 installed (`pip install ultralytics`)  

### 🔹 Steps to Setup

# Clone the repository
git clone https://github.com/Kafoor-A/Trainalyze.git
cd Trainalyze

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
Access the system at: http://localhost:5000

| Feature              | Description                                              |
| :------------------- | :------------------------------------------------------- |
| `/detect`            | Starts YOLO-based exercise detection and motion tracking |
| `/upload`            | Uploads performance data to Firebase                     |
| `/trainer-dashboard` | Displays clients’ progress and workout summaries         |
| `/user-progress`     | Shows personalized workout insights                      |
| `/esp32-data`        | Receives real-time sensor input from ESP32               |

---

🧮 Data Flow Overview

1. Camera Input: Captures gym activity frames

2. YOLOv8 Model: Detects human posture and movement type

3. Flask Server: Processes detection results and sends data

4. Firebase: Stores workout logs, timestamps, and user data

5. Dashboard: Displays analytics for trainers and members

```
🧾 Example Firebase Data
{
  "user_id": "USR123",
  "exercise": "Squats",
  "repetitions": 12,
  "duration": "00:02:35",
  "calories_burned": 25,
  "timestamp": "2025-11-05T18:30:00Z"
}
```
---
📊 Future Enhancements
 - 🧠 Add pose estimation using MediaPipe
 - 📱 Mobile app interface for trainers and users
 - 📤 Export workout history as PDF reports
 - ☁️ Integration with Google Fit / Apple Health
 - 📈 Advanced analytics dashboard using Plotly or Dash

---
🧑‍💻 Developed With
 - Python for backend logic and AI integration
 - YOLOv8 + OpenCV for computer vision
 - Firebase for cloud database and analytics
 - Flask for lightweight web server
 - ESP32 for sensor-based data input

---
🏁 Quick Start

1. Clone the repo (git clone https://github.com/Kafoor-A/Trainalyze.git)
2. Install dependencies (pip install -r requirements.txt)
3. Run the app (python main.py)
4. Open http://localhost:5000
5. Start your gym activity and monitor live analytics

---
📜 License

This project is open-source under the MIT License — you’re free to use, modify, and distribute it with proper attribution.
See the [LICENSE](./LICENSE) file for more details.

---
⭐ Show your support

If you like this project, don’t forget to star 🌟 the repository!

---
**Author:** Abdul Kafoor  
**Department of Electronics and Communication Engineering**  
**Rajalakshmi Engineering College**

