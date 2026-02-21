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
 
 
# --- Load TFLite model ---
clf_interpreter = tf.lite.Interpreter(model_path="facing_classifier.tflite")
clf_interpreter.allocate_tensors()
input_details = clf_interpreter.get_input_details()
output_details = clf_interpreter.get_output_details()
 
THRESH = 0.50  # Threshold to decide facing
 
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
 
# Create tracker for dwell time
tracker = DwellTimeTracker(viewer_threshold=3.0, grace_period=3.0)
 
while True:
    ret, frame = cap.read()
    if not ret:
        break
 
    # Resize frame to model input
    img = cv2.resize(frame, (96, 96))
    input_data = np.expand_dims(img.astype(np.float32)/255.0, axis=0)
 
    # Run inference
    clf_interpreter.set_tensor(input_details[0]['index'], input_data)
    clf_interpreter.invoke()
    prob = clf_interpreter.get_tensor(output_details[0]['index'])[0][0]
 
    # Determine facing (your classifier's output)
    facing = prob > THRESH   # True if "Not Facing"
    is_looking = not facing  # True if "Facing"
 
    # Update dwell-time logic
    state, dwell_time = tracker.update(is_looking)
 
    # Display labels
    label = f"{state} ({dwell_time:.1f}s)"
    color = (0, 255, 0) if state == "Viewer" else (0, 0, 255)
 
    cv2.putText(frame, label, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("MobileNet Facing Detector + Dwell Time", frame)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# Final dwell time when quitting
print(f"Final dwell time: {tracker.get_final_dwell_time():.2f} sec")
 
cap.release()
cv2.destroyAllWindows()