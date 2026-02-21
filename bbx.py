import cv2
import numpy as np
import tensorflow as tf

# Load TFLite model
clf_interpreter = tf.lite.Interpreter(model_path="facing_classifier.tflite")
clf_interpreter.allocate_tensors()
input_details = clf_interpreter.get_input_details()
output_details = clf_interpreter.get_output_details()

THRESH = 0.50  # Threshold for facing
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Haar Cascade face detector (lightweight, fast)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        # Crop and preprocess face
        face_img = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face_img, (96, 96))
        input_data = np.expand_dims(face_resized.astype(np.float32) / 255.0, axis=0)

        # Run classifier
        clf_interpreter.set_tensor(input_details[0]['index'], input_data)
        clf_interpreter.invoke()
        prob = clf_interpreter.get_tensor(output_details[0]['index'])[0][0]

        # Decide facing / not facing
        facing = prob > THRESH
        label = "Facing" if facing else "Not Facing"
        color = (0, 255, 0) if facing else (0, 0, 255)

        # Draw bounding box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{label} ({prob:.2f})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("MobileNet Facing Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
