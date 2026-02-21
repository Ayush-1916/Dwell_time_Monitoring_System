import cv2
import numpy as np
import tensorflow as tf
import time
 
# --- Dwell Time Tracker Class ---
class DwellTimeTracker:
    def __init__(self, viewer_threshold=3.0, grace_period=3.0):
        self.viewer_threshold = viewer_threshold
        self.grace_period = grace_period
 
        self.state = "Bystander"
        self.start_time = None
        self.viewer_start_time = None
        self.look_away_time = None
        self.dwell_time = 0.0
        self.finalized_dwell_time = 0.0
 
    def update(self, is_looking):
        if self.state == "Bystander":
            if is_looking:
                if self.start_time is None:
                    self.start_time = time.time()
                elif time.time() - self.start_time > self.viewer_threshold:
                    self.state = "Viewer"
                    self.viewer_start_time = time.time()
                    self.look_away_time = None
            else:
                self.start_time = None
 
        elif self.state == "Viewer":
            if is_looking:
                if self.viewer_start_time is not None:
                    self.dwell_time += time.time() - self.viewer_start_time
                self.viewer_start_time = time.time()
                self.look_away_time = None
            else:
                if self.look_away_time is None:
                    self.look_away_time = time.time()
                elif time.time() - self.look_away_time > self.grace_period:
                    if self.viewer_start_time is not None:
                        self.dwell_time += time.time() - self.viewer_start_time
                    self.finalized_dwell_time = self.dwell_time
                    self.state = "Bystander"
                    self.start_time = None
                    self.viewer_start_time = None
                    self.look_away_time = None
 
        return self.state, self.dwell_time
 
    def get_final_dwell_time(self):
        if self.state == "Viewer" and self.viewer_start_time is not None:
            self.dwell_time += time.time() - self.viewer_start_time
        return self.dwell_time or self.finalized_dwell_time
 
 
# --- Load TFLite classifier ---
clf_interpreter = tf.lite.Interpreter(model_path="facing_classifier.tflite")
clf_interpreter.allocate_tensors()
input_details = clf_interpreter.get_input_details()
output_details = clf_interpreter.get_output_details()
 
THRESH = 0.50  # Threshold for facing
 
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
 
# Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
 
# One tracker for *all* faces (simple version; can be expanded per face ID)
tracker = DwellTimeTracker(viewer_threshold=3.0, grace_period=3.0)
 
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
        is_looking = facing  # adjust depending on your classifier's output
 
        # Update dwell-time
        state, dwell_time = tracker.update(is_looking)
 
        # Draw bounding box + dwell state
        label = f"{state} ({dwell_time:.1f}s)"
        color = (0, 255, 0) if state == "Viewer" else (0, 0, 255)
 
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
 
    cv2.imshow("Facing Detector + Dwell Time", frame)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# Final dwell time
print(f"Final dwell time: {tracker.get_final_dwell_time():.2f} sec")
 
cap.release()
cv2.destroyAllWindows()
