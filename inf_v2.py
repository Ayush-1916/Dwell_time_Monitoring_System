# import cv2
# import numpy as np
# import tensorflow as tf
# from time import time

# # Load TFLite models
# face_interpreter = tf.lite.Interpreter(model_path="blazeface.tflite")
# face_interpreter.allocate_tensors()
# face_input_details = face_interpreter.get_input_details()
# face_output_details = face_interpreter.get_output_details()

# clf_interpreter = tf.lite.Interpreter(model_path=r"C:\A_Dwell_time\facing_classifier.tflite")
# clf_interpreter.allocate_tensors()
# clf_input_details = clf_interpreter.get_input_details()
# clf_output_details = clf_interpreter.get_output_details()

# # print("Blazeface input:", face_input_details)
# # print("Classifier input:", clf_input_details)


# # Tracker: {face_id: [last_seen_time, total_facing_time, is_facing, bbox]}
# face_tracks = {}
# next_id = 0

# # Thresholds
# FACE_CONF_THRESH = 0.6
# FACING_THRESH = 0.5
# IOU_THRESH = 0.4

# def iou(boxA, boxB):
#     """Compute IoU between two boxes (x1,y1,x2,y2)."""
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[2], boxB[2])
#     yB = min(boxA[3], boxB[3])
#     interArea = max(0, xB - xA) * max(0, yB - yA)
#     boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
#     boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
#     return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     h, w, _ = frame.shape
#     img_resized = cv2.resize(frame, (128, 128))
#     input_data = np.expand_dims(img_resized.astype(np.float32)/255.0, axis=0)

#     # -------- Face detection --------
#     face_interpreter.set_tensor(face_input_details[0]['index'], input_data)
#     face_interpreter.invoke()
#     boxes = face_interpreter.get_tensor(face_output_details[0]['index'])[0]
#     scores = face_interpreter.get_tensor(face_output_details[1]['index'])[0]

#     faces_in_frame = []
#     for box, score in zip(boxes, scores):
#         if score < FACE_CONF_THRESH:
#             continue
#         ymin, xmin, ymax, xmax = box
#         x1, y1 = int(xmin*w), int(ymin*h)
#         x2, y2 = int(xmax*w), int(ymax*h)
#         faces_in_frame.append((x1, y1, x2, y2))

#     # -------- Classification + dwell time --------
#     current_time = time()
#     updated_tracks = {}

#     for (x1, y1, x2, y2) in faces_in_frame:
#         face_crop = frame[y1:y2, x1:x2]
#         if face_crop.size == 0:
#             continue
#         face_crop = cv2.resize(face_crop, (96,96))
#         face_crop = np.expand_dims(face_crop.astype(np.float32)/255.0, axis=0)

#         clf_interpreter.set_tensor(clf_input_details[0]['index'], face_crop)
#         clf_interpreter.invoke()
#         prob = clf_interpreter.get_tensor(clf_output_details[0]['index'])[0][0]

#         facing = prob > FACING_THRESH

#         # Match with existing track via IoU
#         assigned_id = None
#         for face_id, track in face_tracks.items():
#             iou_score = iou((x1,y1,x2,y2), track[3])
#             if iou_score > IOU_THRESH:
#                 assigned_id = face_id
#                 break

#         if assigned_id is None:
#             assigned_id = next_id
#             next_id += 1
#             face_tracks[assigned_id] = [current_time, 0, facing, (x1,y1,x2,y2)]

#         last_time, total_time, _, _ = face_tracks[assigned_id]
#         dt = current_time - last_time
#         if facing:
#             total_time += dt
#         updated_tracks[assigned_id] = [current_time, total_time, facing, (x1,y1,x2,y2)]

#         # Draw
#         cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0) if facing else (0,0,255), 2)
#         cv2.putText(frame, f"{'Facing' if facing else 'Not Facing'} ({total_time:.1f}s)", 
#                     (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

#     face_tracks = updated_tracks

#     cv2.imshow("Facing Detector", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# 

# import cv2
# import numpy as np
# import tensorflow as tf
# from time import time

# # ---------------- Load TFLite Models ----------------
# face_interpreter = tf.lite.Interpreter(model_path="blazeface.tflite")
# face_interpreter.allocate_tensors()
# face_input_details = face_interpreter.get_input_details()
# face_output_details = face_interpreter.get_output_details()

# clf_interpreter = tf.lite.Interpreter(model_path="facing_classifier.tflite")
# clf_interpreter.allocate_tensors()
# clf_input_details = clf_interpreter.get_input_details()
# clf_output_details = clf_interpreter.get_output_details()

# # print("Blazeface input:", face_input_details)
# # print("Classifier input:", clf_input_details)

# # ---------------- Dwell Tracking ----------------
# face_tracks = {}  # {face_id: [last_seen_time, total_facing_time, is_facing]}
# next_id = 0
# THRESH = 0.5  # Facing probability threshold

# # ---------------- Video Capture ----------------
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     h, w, _ = frame.shape

#     # Prepare input for BlazeFace: shape [1, 3, 128, 128]
#     img_resized = cv2.resize(frame, (128, 128))
#     input_data = np.expand_dims(np.transpose(img_resized.astype(np.float32)/255.0, (2,0,1)), axis=0)

#     # -------- Face Detection --------
#     face_interpreter.set_tensor(face_input_details[0]['index'], input_data)
#     face_interpreter.invoke()
#     boxes = face_interpreter.get_tensor(face_output_details[0]['index'])[0]

#     faces_in_frame = []
#     for box in boxes:
#         # BlazeFace outputs [ymin, xmin, ymax, xmax, ...], we take first 4 values
#         if len(box) < 4:
#             continue
#         ymin, xmin, ymax, xmax = box[:4]

#         # Convert normalized coordinates to pixel values
#         x1, y1 = int(xmin * w), int(ymin * h)
#         x2, y2 = int(xmax * w), int(ymax * h)

#         # Ignore invalid boxes
#         if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0 or x2 > w or y2 > h:
#             continue

#         faces_in_frame.append((x1, y1, x2, y2))

#     # -------- Classification + Dwell Time --------
#     current_time = time()
#     for (x1, y1, x2, y2) in faces_in_frame:
#         face_crop = frame[y1:y2, x1:x2]

#         # Check if crop is valid
#         if face_crop.size == 0:
#             continue

#         # Resize for classifier: [1, 96, 96, 3]
#         face_crop = cv2.resize(face_crop, (96, 96))
#         face_crop = np.expand_dims(face_crop.astype(np.float32)/255.0, axis=0)

#         clf_interpreter.set_tensor(clf_input_details[0]['index'], face_crop)
#         clf_interpreter.invoke()
#         prob = clf_interpreter.get_tensor(clf_output_details[0]['index'])[0][0]

#         facing = prob > THRESH
#         cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0) if facing else (0,0,255), 2)
#         cv2.putText(frame, f"{'Facing' if facing else 'Not Facing'}", (x1, y1-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

#         # Track dwell time
#         face_id = next_id
#         if face_id not in face_tracks:
#             face_tracks[face_id] = [current_time, 0, facing]  # [last_seen_time, total_facing_time, is_facing]

#         if facing:
#             face_tracks[face_id][1] += 1/30  # approximate seconds per frame (30 FPS)
#         face_tracks[face_id][0] = current_time
#         face_tracks[face_id][2] = facing
#         next_id += 1

#     cv2.imshow("Facing Detector", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
# import tensorflow as tf
# from time import time

# # Load anchors
# anchors = np.load("anchors.npy")

# # Load TFLite models
# face_interpreter = tf.lite.Interpreter(model_path="blazeface.tflite")
# face_interpreter.allocate_tensors()
# face_input_details = face_interpreter.get_input_details()
# face_output_details = face_interpreter.get_output_details()

# clf_interpreter = tf.lite.Interpreter(model_path="facing_classifier.tflite")
# clf_interpreter.allocate_tensors()
# clf_input_details = clf_interpreter.get_input_details()
# clf_output_details = clf_interpreter.get_output_details()

# THRESH = 0.5
# face_tracks = {}
# next_id = 0

# cap = cv2.VideoCapture(0)

# def decode_boxes(raw_boxes, anchors, input_size=(128,128)):
#     """Decode TFLite BlazeFace output boxes using anchors"""
#     # raw_boxes: [896, 16], anchors: [896, 4]
#     boxes = []
#     scores = []
#     for i in range(len(anchors)):
#         score = raw_boxes[i, 0]  # objectness probability
#         if score > 0.5:
#             # Decode relative box coordinates
#             cx = raw_boxes[i, 1] + anchors[i, 0]
#             cy = raw_boxes[i, 2] + anchors[i, 1]
#             w = raw_boxes[i, 3]
#             h = raw_boxes[i, 4]
#             xmin = max(0, cx - w/2)
#             ymin = max(0, cy - h/2)
#             xmax = min(1, cx + w/2)
#             ymax = min(1, cy + h/2)
#             boxes.append([ymin, xmin, ymax, xmax])
#             scores.append(score)
#     return boxes, scores

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     h, w, _ = frame.shape
#     img_resized = cv2.resize(frame, (128, 128))
#     input_data = np.expand_dims(np.transpose(img_resized.astype(np.float32)/255.0, (2,0,1)), axis=0)

#     # BlazeFace inference
#     face_interpreter.set_tensor(face_input_details[0]['index'], input_data)
#     face_interpreter.invoke()
#     raw_boxes = face_interpreter.get_tensor(face_output_details[0]['index'])[0]  # shape [896,16]

#     boxes, scores = decode_boxes(raw_boxes, anchors)

#     faces_in_frame = []
#     for box in boxes:
#         ymin, xmin, ymax, xmax = box
#         x1, y1 = int(xmin*w), int(ymin*h)
#         x2, y2 = int(xmax*w), int(ymax*h)
#         if x2 <= x1 or y2 <= y1:
#             continue
#         faces_in_frame.append((x1,y1,x2,y2))

#     current_time = time()
#     for (x1,y1,x2,y2) in faces_in_frame:
#         face_crop = frame[y1:y2, x1:x2]
#         if face_crop.size == 0:
#             continue
#         face_crop = cv2.resize(face_crop, (96,96))
#         face_crop = np.expand_dims(face_crop.astype(np.float32)/255.0, axis=0)

#         clf_interpreter.set_tensor(clf_input_details[0]['index'], face_crop)
#         clf_interpreter.invoke()
#         prob = clf_interpreter.get_tensor(clf_output_details[0]['index'])[0][0]

#         facing = prob > THRESH
#         cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0) if facing else (0,0,255), 2)
#         cv2.putText(frame, f"{'Facing' if facing else 'Not Facing'}", (x1, y1-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

#         face_id = next_id
#         if face_id not in face_tracks:
#             face_tracks[face_id] = [current_time, 0, facing]
#         if facing:
#             face_tracks[face_id][1] += 1/30
#         face_tracks[face_id][0] = current_time
#         face_tracks[face_id][2] = facing
#         next_id += 1

#     cv2.imshow("Facing Detector", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# import cv2
# import numpy as np
# import tensorflow as tf
# from time import time

# # Load BlazeFace TFLite model
# face_interpreter = tf.lite.Interpreter(model_path="blazeface.tflite")
# face_interpreter.allocate_tensors()
# face_input_details = face_interpreter.get_input_details()
# face_output_details = face_interpreter.get_output_details()

# # Load Facing Classifier TFLite model
# clf_interpreter = tf.lite.Interpreter(model_path="facing_classifier.tflite")
# clf_interpreter.allocate_tensors()
# clf_input_details = clf_interpreter.get_input_details()
# clf_output_details = clf_interpreter.get_output_details()

# THRESH = 0.5  # probability threshold for facing

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     h, w, _ = frame.shape
#     # Prepare input for BlazeFace (channel-first)
#     img_input = cv2.resize(frame, (128, 128))
#     img_input = np.transpose(img_input, (2, 0, 1))  # HWC -> CHW
#     img_input = np.expand_dims(img_input.astype(np.float32) / 255.0, axis=0)
#     face_interpreter.set_tensor(face_input_details[0]['index'], img_input)
#     face_interpreter.invoke()

#     # Get outputs
#     boxes = face_interpreter.get_tensor(face_output_details[0]['index'])[0]  # shape: [N,4] ?
#     scores = face_interpreter.get_tensor(face_output_details[1]['index'])[0]  # shape: [N] ?

#     faces_in_frame = []
#     for i, score in enumerate(scores):
#         if score < 0.5:
#             continue
#         box = boxes[i]
#         # box may be [ymin, xmin, ymax, xmax] normalized
#         ymin = int(box[0] * h)
#         xmin = int(box[1] * w)
#         ymax = int(box[2] * h)
#         xmax = int(box[3] * w)
#         faces_in_frame.append((xmin, ymin, xmax, ymax))

#     # Classification
#     for (x1, y1, x2, y2) in faces_in_frame:
#         face_crop = frame[y1:y2, x1:x2]
#         if face_crop.size == 0:
#             continue
#         face_crop = cv2.resize(face_crop, (96, 96))
#         face_input = np.expand_dims(face_crop.astype(np.float32)/255.0, axis=0)

#         clf_interpreter.set_tensor(clf_input_details[0]['index'], face_input)
#         clf_interpreter.invoke()
#         prob = clf_interpreter.get_tensor(clf_output_details[0]['index'])[0][0]

#         facing = prob > THRESH
#         color = (0,255,0) if facing else (0,0,255)
#         cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
#         cv2.putText(frame, f"{'Facing' if facing else 'Not Facing'}",
#                     (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

#     cv2.imshow("Facing Detector", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


import cv2
import numpy as np
import tensorflow as tf
from time import time

# ---------------- Load TFLite models ----------------
face_interpreter = tf.lite.Interpreter(model_path="blazeface.tflite")
face_interpreter.allocate_tensors()
face_input_details = face_interpreter.get_input_details()
face_output_details = face_interpreter.get_output_details()

clf_interpreter = tf.lite.Interpreter(model_path=r"C:\A_Dwell_time\facing_classifier.tflite")
clf_interpreter.allocate_tensors()
clf_input_details = clf_interpreter.get_input_details()
clf_output_details = clf_interpreter.get_output_details()

# ---------------- Tracker ----------------
# {face_id: [last_seen_time, total_facing_time, box, is_facing]}
face_tracks = {}
next_id = 0
THRESH = 0.5
MAX_MISSING = 0.5  # seconds to keep last box if face missing

# ---------------- Webcam ----------------
cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    img_resized = cv2.resize(frame, (128, 128))
    input_data = np.expand_dims(np.transpose(img_resized.astype(np.float32)/255.0, (2,0,1)), axis=0)

    # -------- Face detection --------
    face_interpreter.set_tensor(face_input_details[0]['index'], input_data)
    face_interpreter.invoke()

    boxes = face_interpreter.get_tensor(face_output_details[0]['index'])[0]  # [num_anchors,4]
    scores = face_interpreter.get_tensor(face_output_details[1]['index'])[0]  # confidence

    faces_in_frame = []
    for i, score in enumerate(scores):
        if score < 0.5:
            continue
        box = boxes[i]
        ymin = int(box[0] * h)
        xmin = int(box[1] * w)
        ymax = int(box[2] * h)
        xmax = int(box[3] * w)

        if xmax - xmin <= 0 or ymax - ymin <= 0:
            continue
        faces_in_frame.append((xmin, ymin, xmax, ymax))

    # -------- Classification + dwell time --------
    current_time = time()
    updated_ids = []

    for (x1, y1, x2, y2) in faces_in_frame:
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue
        face_crop = cv2.resize(face_crop, (96,96))
        face_crop = np.expand_dims(face_crop.astype(np.float32)/255.0, axis=0)

        clf_interpreter.set_tensor(clf_input_details[0]['index'], face_crop)
        clf_interpreter.invoke()
        prob = clf_interpreter.get_tensor(clf_output_details[0]['index'])[0][0]

        facing = prob > THRESH

        # Assign new face ID
        face_id = next_id
        next_id += 1
        updated_ids.append(face_id)

        # Update tracker
        face_tracks[face_id] = [current_time, facing*1/30, (x1,y1,x2,y2), facing]

    # Remove old faces
    to_delete = []
    for fid, (last_seen, total, box, is_facing) in face_tracks.items():
        if current_time - last_seen > MAX_MISSING:
            to_delete.append(fid)
    for fid in to_delete:
        del face_tracks[fid]

    # -------- Draw boxes --------
    for fid, (last_seen, total, box, is_facing) in face_tracks.items():
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0) if is_facing else (0,0,255), 2)
        cv2.putText(frame, f"{'Facing' if is_facing else 'Not Facing'} {total:.1f}s", 
                    (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imshow("Facing Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# import cv2
# import numpy as np
# import tensorflow as tf
# from time import time

# # ---------------- Load TFLite models ----------------
# face_interpreter = tf.lite.Interpreter(model_path="blazeface.tflite")
# face_interpreter.allocate_tensors()
# face_input_details = face_interpreter.get_input_details()
# face_output_details = face_interpreter.get_output_details()

# clf_interpreter = tf.lite.Interpreter(model_path="facing_classifier.tflite")
# clf_interpreter.allocate_tensors()
# clf_input_details = clf_interpreter.get_input_details()
# clf_output_details = clf_interpreter.get_output_details()

# # ---------------- Parameters ----------------
# CONF_THRESH = 0.3   # minimum confidence for detection
# TOP_K = 1           # max faces per frame
# THRESH = 0.3        # facing probability threshold

# # ---------------- Simple tracker ----------------
# face_tracks = {}    # {face_id: [last_seen_time, total_facing_time, is_facing]}
# next_id = 0

# # ---------------- Webcam ----------------
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     h, w, _ = frame.shape
#     img_resized = cv2.resize(frame, (128, 128))
#     input_data = np.expand_dims(np.transpose(img_resized.astype(np.float32)/255.0, (2,0,1)), axis=0)

#     # -------- Face detection --------
#     face_interpreter.set_tensor(face_input_details[0]['index'], input_data)
#     face_interpreter.invoke()
#     boxes = face_interpreter.get_tensor(face_output_details[0]['index'])[0]
#     scores = face_interpreter.get_tensor(face_output_details[1]['index'])[0]

#     # Filter by confidence
#     filtered = [(box, score) for box, score in zip(boxes, scores) if score > CONF_THRESH]
#     filtered = sorted(filtered, key=lambda x: x[1], reverse=True)[:TOP_K]

#     faces_in_frame = []
#     for box, score in filtered:
#         ymin, xmin, ymax, xmax = box[:4]
#         if (ymax - ymin) <= 0 or (xmax - xmin) <= 0:
#             continue
#         x1, y1 = int(xmin*w), int(ymin*h)
#         x2, y2 = int(xmax*w), int(ymax*h)
#         faces_in_frame.append((x1, y1, x2, y2))

#     # -------- Classification + dwell time --------
#     current_time = time()
#     for (x1, y1, x2, y2) in faces_in_frame:
#         face_crop = frame[y1:y2, x1:x2]
#         if face_crop.size == 0:
#             continue
#         face_crop = cv2.resize(face_crop, (96,96))
#         face_crop = np.expand_dims(face_crop.astype(np.float32)/255.0, axis=0)

#         clf_interpreter.set_tensor(clf_input_details[0]['index'], face_crop)
#         clf_interpreter.invoke()
#         prob = clf_interpreter.get_tensor(clf_output_details[0]['index'])[0][0]

#         facing = prob > THRESH
#         cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0) if facing else (0,0,255), 2)
#         cv2.putText(frame, f"{'Facing' if facing else 'Not Facing'}", (x1, y1-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

#         # Simple dwell tracking
#         face_id = next_id
#         if face_id not in face_tracks:
#             face_tracks[face_id] = [current_time, 0, facing]

#         if facing:
#             face_tracks[face_id][1] += 1/30   # approx seconds per frame
#         face_tracks[face_id][0] = current_time
#         face_tracks[face_id][2] = facing
#         next_id += 1

#     cv2.imshow("Facing Detector", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

