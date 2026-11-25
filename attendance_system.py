import cv2
import csv
import os
from datetime import datetime

MODEL_PATH = "data/trainer.yml"
LABEL_PATH = "data/labels.txt"
ATTENDANCE_FILE = "attendance/attendance.csv"

os.makedirs("attendance", exist_ok=True)

if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Roll Number", "Date", "Time", "Status"])

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)

labels = {}

with open(LABEL_PATH, "r") as file:
    for line in file:
        line = line.strip()

        if line:
            label, roll = line.split(",")
            labels[int(label)] = roll

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

marked_today = set()
today = datetime.now().strftime("%Y-%m-%d")

if os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, "r") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row["Date"] == today:
                marked_today.add(row["Roll Number"])


def mark_attendance(roll):
    today = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")

    if roll in marked_today:
        return False

    with open(ATTENDANCE_FILE, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([roll, today, current_time, "Present"])

    marked_today.add(roll)

    print(f"Attendance marked: {roll}")

    return True


camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("ERROR: Camera open nahi ho raha!")
    exit()

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

WINDOW_NAME = "Face Recognition Attendance System"

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 900, 650)

print("================================")
print("ATTENDANCE SYSTEM STARTED")
print("================================")
print("Camera window should open now")
print("Press Q inside the camera window to exit")


while True:
    ret, frame = camera.read()

    if not ret:
        print("ERROR: Camera frame read nahi ho raha!")
        break

    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]

        try:
            label, confidence = recognizer.predict(face)
        except Exception:
            continue

        if confidence < 70 and label in labels:
            roll = labels[label]

            mark_attendance(roll)

            text = f"{roll} - PRESENT"

        else:
            text = "UNKNOWN"

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow(WINDOW_NAME, frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == ord("Q"):
        print("Q pressed. Closing system...")
        break


camera.release()
cv2.destroyAllWindows()

print("Attendance System Closed.")
