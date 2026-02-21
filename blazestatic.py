# import cv2
# import numpy as np
# import tensorflow as tf

# # ---------------- Load BlazeFace TFLite model ----------------
# face_interpreter = tf.lite.Interpreter(model_path="blazeface.tflite")
# face_interpreter.allocate_tensors()
# face_input_details = face_interpreter.get_input_details()
# face_output_details = face_interpreter.get_output_details()

# # ---------------- Load test image ----------------
# img_path = r"C:\A_Dwell_time\dataset\train\facing\000001.jpg"  # Replace with your image path
# img = cv2.imread(img_path)
# if img is None:
#     raise ValueError("Image not found or path is incorrect")

# h, w, _ = img.shape

# # Resize image to model input
# input_h, input_w = face_input_details[0]['shape'][2:4]  # BlazeFace expects NHWC or NCHW? Check shape
# input_data = cv2.resize(img, (input_w, input_h))
# # BlazeFace TFLite expects CHW (3, 128, 128), so transpose if needed
# if face_input_details[0]['shape'][1] == 3:  # NCHW
#     input_data = np.transpose(input_data, (2, 0, 1))
# input_data = np.expand_dims(input_data.astype(np.float32)/255.0, axis=0)

# # ---------------- Run inference ----------------
# face_interpreter.set_tensor(face_input_details[0]['index'], input_data)
# face_interpreter.invoke()

# # Get bounding boxes
# boxes = face_interpreter.get_tensor(face_output_details[0]['index'])[0]

# # THRESH = 0.5
# # for i, box in enumerate(boxes):
# #     if scores[i] < THRESH:
# #         continue
# #     ymin, xmin, ymax, xmax = box[:4]
# #     x1, y1 = int(xmin*w), int(ymin*h)
# #     x2, y2 = int(xmax*w), int(ymax*h)
# #     cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)

# for box in boxes:
#     if np.all(box == 0):  # Skip empty detections
#         continue
#     ymin, xmin, ymax, xmax = box[:4]  # take only first 4 elements
#     x1, y1 = int(xmin * w), int(ymin * h)
#     x2, y2 = int(xmax * w), int(ymax * h)
#     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# # Show image with detected faces
# cv2.imshow("BlazeFace Detection", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import numpy as np
import tensorflow as tf

# -------- Load models --------
face_interpreter = tf.lite.Interpreter(model_path="blazeface.tflite")
face_interpreter.allocate_tensors()
face_input_details = face_interpreter.get_input_details()
face_output_details = face_interpreter.get_output_details()

clf_interpreter = tf.lite.Interpreter(model_path="facing_classifier.tflite")
clf_interpreter.allocate_tensors()
clf_input_details = clf_interpreter.get_input_details()
clf_output_details = clf_interpreter.get_output_details()

# -------- Settings --------
CONF_THRESHOLD = 0.5  # confidence threshold for BlazeFace
CLF_THRESHOLD = 0.5   # threshold for facing classifier

# -------- Load test image --------
img = cv2.imread("test_image.jpg")  # replace with your image path
h, w, _ = img.shape

# Prepare input for BlazeFace
img_resized = cv2.resize(img, (128, 128))
input_data = np.expand_dims(img_resized.astype(np.float32)/255.0, axis=0)

# -------- Face detection --------
face_interpreter.set_tensor(face_input_details[0]['index'], input_data)
face_interpreter.invoke()

# BlazeFace outputs
boxes = face_interpreter.get_tensor(face_output_details[0]['index'])[0]
scores = face_interpreter.get_tensor(face_output_details[1]['index'])[0]

faces_in_image = []
for i, box in enumerate(boxes):
    if scores[i] < CONF_THRESHOLD:
        continue  # skip low-confidence boxes
    ymin, xmin, ymax, xmax = box[:4]
    x1, y1 = int(xmin*w), int(ymin*h)
    x2, y2 = int(xmax*w), int(ymax*h)
    faces_in_image.append((x1, y1, x2, y2))

# -------- Classification + display --------
for (x1, y1, x2, y2) in faces_in_image:
    face_crop = img[y1:y2, x1:x2]
    if face_crop.size == 0:  # skip empty crops
        continue
    face_crop = cv2.resize(face_crop, (96, 96))
    face_crop = np.expand_dims(face_crop.astype(np.float32)/255.0, axis=0)

    clf_interpreter.set_tensor(clf_input_details[0]['index'], face_crop)
    clf_interpreter.invoke()
    prob = clf_interpreter.get_tensor(clf_output_details[0]['index'])[0][0]
    facing = prob > CLF_THRESHOLD

    color = (0, 255, 0) if facing else (0, 0, 255)
    label = "Facing" if facing else "Not Facing"
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

cv2.imshow("Face Detection + Classification", img)
cv2.waitKey(0)
cv2.destroyAllWindows()



