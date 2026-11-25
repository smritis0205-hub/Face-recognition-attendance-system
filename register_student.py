import cv2
import os

# Student details
name = input("Enter student name: ").strip()
roll = input("Enter roll number: ").strip()

if not name or not roll:
    print("Name aur Roll Number required hai!")
    exit()

# Create student folder
student_folder = os.path.join("data", "students", roll)
os.makedirs(student_folder, exist_ok=True)

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera open nahi ho raha!")
    exit()

count = 0

print("\nCamera started!")
print("Face camera ke saamne rakho.")
print("20 face images capture hongi.")
print("Press Q to cancel.\n")

while True:
    ret, frame = camera.read()

    if not ret:
        print("Camera frame nahi mil raha!")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    for (x, y, w, h) in faces:

        # Face rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Save face
        count += 1

        face = gray[y:y+h, x:x+w]

        filename = os.path.join(
            student_folder,
            f"{roll}_{count}.jpg"
        )

        cv2.imwrite(filename, face)

        cv2.putText(
            frame,
            f"Captured: {count}/20",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow("Student Registration", frame)

    if cv2.waitKey(100) & 0xFF == ord("q"):
        break

    if count >= 20:
        break

camera.release()
cv2.destroyAllWindows()

print("\nRegistration completed!")
print(f"Name: {name}")
print(f"Roll Number: {roll}")
print(f"Images saved: {count}")