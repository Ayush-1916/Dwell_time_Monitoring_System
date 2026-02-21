# dwell_time_mediapipe.py
import cv2
import time
import math
import numpy as np
import mediapipe as mp

# ----------------------------
# Config
# ----------------------------
MAX_FACES = 5
MIN_DET_CONF = 0.5
MIN_TRACK_CONF = 0.5

# Facing thresholds (tune these)
YAW_THRESH_DEG = 15.0   # left-right
PITCH_THRESH_DEG = 12.0 # up-down

# Smoothing factor for angles (EMA)
EMA_ALPHA = 0.4

# Minimum continuous seconds considered a valid dwell (avoid noise)
MIN_CONTIGUOUS_SECONDS = 0.3

# Grace period (seconds) to avoid immediate reset on short occlusion
GRACE_PERIOD = 0.6

# ----------------------------
# 3D model points for solvePnP (approx, in mm)
# Use the same number/order as the 2D image_points extracted below
# ----------------------------
MODEL_3D_POINTS = np.array([
    (0.0, 0.0, 0.0),            # Nose tip
    (0.0, -330.0, -65.0),       # Chin
    (-225.0, 170.0, -135.0),    # Left eye left corner
    (225.0, 170.0, -135.0),     # Right eye right corner
    (-150.0, -150.0, -125.0),   # Left Mouth corner
    (150.0, -150.0, -125.0)     # Right mouth corner
], dtype=np.float64)

# ----------------------------
# Mediapipe landmark indices (FaceMesh)
# These are commonly used indices: adjust if you want different points
# ----------------------------
LANDMARK_IDS = {
    'nose_tip': 1,
    'chin': 152,
    'left_eye_outer': 33,
    'right_eye_outer': 263,
    'left_mouth': 61,
    'right_mouth': 291
}

# ----------------------------
# Helpers
# ----------------------------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False,
                            max_num_faces=MAX_FACES,
                            refine_landmarks=False,
                            min_detection_confidence=MIN_DET_CONF,
                            min_tracking_confidence=MIN_TRACK_CONF)

def rotation_matrix_to_euler_angles(R):
    # From rotation matrix R -> Euler angles (x = pitch, y = yaw, z = roll)
    sy = math.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.degrees(x), np.degrees(y), np.degrees(z)  # pitch, yaw, roll

# ----------------------------
# Dwell Timer state
# ----------------------------
accumulated_dwell = 0.0
current_start = None
last_facing_state = False
last_face_time = 0.0  # last time we saw a face (for grace period)
prev_yaw, prev_pitch = 0.0, 0.0

# ----------------------------
# Main loop
# ----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Change camera index if needed.")

print("Press 'q' to quit. Starting camera...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    face_detected = False
    facing = False
    yaw_deg = 0.0
    pitch_deg = 0.0
    roll_deg = 0.0

    if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
        lm = results.multi_face_landmarks[0].landmark
        # build 2D image points
        try:
            image_points = np.array([
                (lm[LANDMARK_IDS['nose_tip']].x * w, lm[LANDMARK_IDS['nose_tip']].y * h),  # nose
                (lm[LANDMARK_IDS['chin']].x * w, lm[LANDMARK_IDS['chin']].y * h),        # chin
                (lm[LANDMARK_IDS['left_eye_outer']].x * w, lm[LANDMARK_IDS['left_eye_outer']].y * h),  # left eye
                (lm[LANDMARK_IDS['right_eye_outer']].x * w, lm[LANDMARK_IDS['right_eye_outer']].y * h),# right eye
                (lm[LANDMARK_IDS['left_mouth']].x * w, lm[LANDMARK_IDS['left_mouth']].y * h),          # left mouth
                (lm[LANDMARK_IDS['right_mouth']].x * w, lm[LANDMARK_IDS['right_mouth']].y * h)         # right mouth
            ], dtype=np.float64)

            # camera matrix (approx)
            focal_length = w
            center = (w / 2.0, h / 2.0)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)
            dist_coeffs = np.zeros((4,1))

            success, rvec, tvec = cv2.solvePnP(MODEL_3D_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            if success:
                R, _ = cv2.Rodrigues(rvec)
                pitch_deg, yaw_deg, roll_deg = rotation_matrix_to_euler_angles(R)

                # smoothing
                smoothed_yaw = EMA_ALPHA * yaw_deg + (1 - EMA_ALPHA) * prev_yaw
                smoothed_pitch = EMA_ALPHA * pitch_deg + (1 - EMA_ALPHA) * prev_pitch
                prev_yaw, prev_pitch = smoothed_yaw, smoothed_pitch

                # facing decision
                facing = (abs(smoothed_yaw) <= YAW_THRESH_DEG) and (abs(smoothed_pitch) <= PITCH_THRESH_DEG)
                face_detected = True
                last_face_time = time.time()

                # draw some markers
                for (x,y) in image_points.astype(int):
                    cv2.circle(frame, (int(x), int(y)), 2, (0,255,0), -1)

                # draw euler text
                cv2.putText(frame, f"Yaw: {smoothed_yaw:.1f} Pitch: {smoothed_pitch:.1f} Roll: {roll_deg:.1f}",
                            (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        except Exception as e:
            # if any landmark not found, skip
            print("Landmark extraction failed:", e)

    # Timer logic with grace period
    now = time.time()
    if facing:
        if not last_facing_state:
            # just started facing
            current_start = now
        # while facing, we don't finalize anything (we'll compute running dwell shown live)
    else:
        # no face facing now
        if last_facing_state:
            # just stopped facing: compute contiguous duration
            elapsed = now - (current_start or now)
            # only add if contiguous > MIN_CONTIGUOUS_SECONDS
            if elapsed >= MIN_CONTIGUOUS_SECONDS:
                accumulated_dwell += elapsed
            current_start = None
        else:
            # not facing and was not facing earlier; also apply grace period to avoid resetting on short occlusion
            if not face_detected and (now - last_face_time) <= GRACE_PERIOD:
                # treat it as still facing for the grace period (do nothing)
                pass

    # compute running dwell time to display
    running = accumulated_dwell
    if facing and current_start is not None:
        running += (now - current_start)

    # overlay dwell time
    cv2.putText(frame, f"Dwell (s): {running:.2f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,200,255), 2)
    cv2.putText(frame, f"Facing: {facing}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

    # update state
    last_facing_state = facing

    cv2.imshow("DwellTime (MediaPipe)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# when finished, if user was still facing, add leftover
if last_facing_state and current_start is not None:
    leftover = time.time() - current_start
    if leftover >= MIN_CONTIGUOUS_SECONDS:
        accumulated_dwell += leftover

print(f"Total dwell seconds: {accumulated_dwell:.2f}")
cap.release()
cv2.destroyAllWindows()
