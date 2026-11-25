import cv2
import os
import numpy as np

DATASET_PATH = "data/students"
MODEL_PATH = "data/trainer.yml"

faces = []
labels = []
label_to_name = {}

current_label = 0

for roll in os.listdir(DATASET_PATH):

    student_path = os.path.join(DATASET_PATH, roll)

    if not os.path.isdir(student_path):
        continue

    label_to_name[current_label] = roll

    for image_name in os.listdir(student_path):

        image_path = os.path.join(student_path, image_name)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            continue

        faces.append(image)
        labels.append(current_label)

    current_label += 1


if not faces:
    print("No student images found!")
    exit()

print(f"Training on {len(faces)} images...")

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.train(
    faces,
    np.array(labels)
)

recognizer.write(MODEL_PATH)

# Save label information
with open("data/labels.txt", "w") as file:
    for label, roll in label_to_name.items():
        file.write(f"{label},{roll}\n")

print("\nTraining completed!")
print("Model saved:", MODEL_PATH)
print("Labels saved: data/labels.txt")