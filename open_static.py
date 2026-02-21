# import cv2
# import numpy as np
# import tensorflow as tf

# # Load YuNet ONNX
# net = cv2.dnn.readNetFromONNX("face_detection_yunet_2023mar.onnx")

# # Load classifier
# clf_interpreter = tf.lite.Interpreter(model_path="facing_classifier.tflite")
# clf_interpreter.allocate_tensors()
# input_details = clf_interpreter.get_input_details()
# output_details = clf_interpreter.get_output_details()

# THRESH = 0.5

# img = cv2.imread(r"C:\A_Dwell_time\dataset\train\facing\000001.jpg")
# h, w, _ = img.shape

# # Prepare input blob for YuNet
# # blob = cv2.dnn.blobFromImage(img, 1.0, (320, 320), [0, 0, 0], swapRB=True, crop=False)
# # net.setInput(blob)
# # detections = net.forward()[0][0]  # [num_faces, 16]
# blob = cv2.dnn.blobFromImage(img, 1.0, (320, 320), [104, 117, 123], True, False)
# net.setInput(blob)
# detections = net.forward()

# if len(detections.shape) != 4:
#     print("Unexpected output shape, skipping frame")
    

# # detections = net.forward()
# # print(detections.shape)
# # print(detections)


# for det in detections[0,0]:
#     score = det[14]
#     if score < 0.5:
#         continue
    
#     x, y, w_box, h_box = int(det[0]*w), int(det[1]*h), int(det[2]*w), int(det[3]*h)
#     face_crop = img[y:y+h_box, x:x+w_box]
#     if face_crop.size == 0:
#         continue
    
#     # Preprocess for classifier
#     face_resized = cv2.resize(face_crop, (96, 96))
#     input_data = np.expand_dims(face_resized.astype(np.float32)/255.0, axis=0)
    
#     # Run classifier
#     clf_interpreter.set_tensor(input_details[0]['index'], input_data)
#     clf_interpreter.invoke()
#     prob = clf_interpreter.get_tensor(output_details[0]['index'])[0][0]
    
#     facing = prob > THRESH
#     label = "Facing" if facing else "Not Facing"
#     color = (0, 255, 0) if facing else (0, 0, 255)
    
#     cv2.rectangle(img, (x, y), (x+w_box, y+h_box), color, 2)
#     cv2.putText(img, f"{label} ({prob:.2f})", (x, y-10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# cv2.imshow("YuNet ONNX + Classifier", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import numpy as np
import tensorflow as tf

# =======================
# Load TFLite facing classifier
# =======================
clf_interpreter = tf.lite.Interpreter(model_path="facing_classifier.tflite")
clf_interpreter.allocate_tensors()
input_details = clf_interpreter.get_input_details()
output_details = clf_interpreter.get_output_details()
THRESH = 0.95  # classifier threshold

# =======================
# Load YuNet face detector
# =======================
face_net = cv2.FaceDetectorYN.create(
    "face_detection_yunet_2023mar.onnx", "", (320, 320),
    score_threshold=0.5, nms_threshold=0.3, top_k=50
)

# =======================
# Start webcam
# =======================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    face_net.setInputSize((w, h))
    faces = face_net.detect(frame)[1]  # returns faces or None

    if faces is not None:
        for face in faces:
            # face format: [x, y, width, height, score, landmarks...]
            x, y, bw, bh = [int(v) for v in face[:4]]
            score = face[4]

            if score < 0.5:
                continue

            # Crop face
            face_crop = frame[y:y+bh, x:x+bw]
            if face_crop.size == 0:
                continue

            # Preprocess for classifier
            face_resized = cv2.resize(face_crop, (96, 96))
            input_data = np.expand_dims(face_resized.astype(np.float32)/255.0, axis=0)

            # Run classifier
            clf_interpreter.set_tensor(input_details[0]['index'], input_data)
            clf_interpreter.invoke()
            prob = clf_interpreter.get_tensor(output_details[0]['index'])[0][0]

            # Facing / Not Facing
            facing = prob > THRESH
            label = "not Facing" if facing else "Facing"
            color = (0, 0, 255) if facing else (0, 255, 0)

            # Draw results
            cv2.rectangle(frame, (x, y), (x+bw, y+bh), color, 2)
            cv2.putText(frame, f"{label} ({prob:.2f})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("YuNet + MobileNetV2 Classifier", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
