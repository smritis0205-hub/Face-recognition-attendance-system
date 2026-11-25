import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera open nahi ho raha!")
    exit()

print("Camera started!")
print("Press Q to quit.")

while True:
    ret, frame = camera.read()

    if not ret:
        print("Frame read nahi ho raha!")
        break

    cv2.imshow("Face Attendance - Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()