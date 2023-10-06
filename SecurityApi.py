import cv2
import datetime

face_cascade = cv2.CascadeClassifier('C:\\Users\\theve\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None
recording = False

while True:
    ret, frame = cap.read()

    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        if not recording:
            now = datetime.datetime.now()
            timestamp = now.strftime("%Y-%m-%d_%I-%M-%S%p")
            filename = f"record_{timestamp}.avi"
            recording = True
            out = cv2.VideoWriter(filename, fourcc, 60.0, (640, 480))
    else:
        if recording:
            out.release()
            recording = False

    if recording:
        out.write(frame)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Face Detection and Recording', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if recording:
    out.release()
cv2.destroyAllWindows()