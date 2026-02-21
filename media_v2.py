# import cv2
# import numpy as np
# import mediapipe as mp
# import tensorflow as tf

# # Load TFLite facing classifier
# clf_interpreter = tf.lite.Interpreter(model_path="facing_classifier.tflite")
# clf_interpreter.allocate_tensors()
# input_details = clf_interpreter.get_input_details()
# output_details = clf_interpreter.get_output_details()

# THRESH = 0.90  # classifier threshold

# # Mediapipe face detector
# mp_face = mp.solutions.face_detection

# cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.60) as face_detection:
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Convert to RGB (Mediapipe requires RGB)
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = face_detection.process(rgb)

#         if results.detections:
#             for detection in results.detections:
#                 bboxC = detection.location_data.relative_bounding_box
#                 h, w, _ = frame.shape
#                 x, y, bw, bh = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

#                 # Crop the face
#                 face_crop = frame[y:y+bh, x:x+bw]
#                 if face_crop.size == 0:
#                     continue

#                 # Preprocess for classifier
#                 face_resized = cv2.resize(face_crop, (96, 96))
#                 input_data = np.expand_dims(face_resized.astype(np.float32) / 255.0, axis=0)

#                 # Run classifier
#                 clf_interpreter.set_tensor(input_details[0]['index'], input_data)
#                 clf_interpreter.invoke()
#                 prob = clf_interpreter.get_tensor(output_details[0]['index'])[0][0]

#                 # Facing / Not Facing
#                 facing = prob > THRESH
#                 label = "Facing" if facing else "Not Facing"
#                 color = (0, 255, 0) if facing else (0, 0, 255)

#                 # Draw results
#                 cv2.rectangle(frame, (x, y), (x+bw, y+bh), color, 2)
#                 cv2.putText(frame, f"{label} ({prob:.2f})", (x, y - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#         cv2.imshow("Mediapipe + MobileNetV2 Classifier", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# cap.release()
# cv2.destroyAllWindows()


import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Load TFLite facing classifier
clf_interpreter = tf.lite.Interpreter(model_path="facing_classifier.tflite")
clf_interpreter.allocate_tensors()
input_details = clf_interpreter.get_input_details()
output_details = clf_interpreter.get_output_details()

THRESH = 0.7  # classifier threshold

# Mediapipe face detector
mp_face = mp.solutions.face_detection

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # try 0 if 1 doesn’t work

with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.60) as face_detection:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB (Mediapipe requires RGB)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb)

        if results.detections:
            h, w, _ = frame.shape
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x, y, bw, bh = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

                # Clip bounding box to frame boundaries
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(w, x + bw), min(h, y + bh)

                # Crop the face safely
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                # Preprocess for classifier
                face_resized = cv2.resize(face_crop, (96, 96))
                face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)  # convert to RGB
                input_data = np.expand_dims(face_rgb.astype(np.float32) / 255.0, axis=0)  # normalize to [0,1]

                # Run classifier
                clf_interpreter.set_tensor(input_details[0]['index'], input_data)
                clf_interpreter.invoke()
                prob = clf_interpreter.get_tensor(output_details[0]['index'])[0][0]

                # Facing / Not Facing
                facing = prob > THRESH
                label = "Facing" if facing else "Not Facing"
                color = (0, 255, 0) if facing else (0, 0, 255)

                # Draw results
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} ({prob:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Mediapipe + MobileNetV2 Classifier", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
