import cv2
import numpy as np
import tensorflow as tf

# Load TFLite model
clf_interpreter = tf.lite.Interpreter(model_path="facing_classifier.tflite")
clf_interpreter.allocate_tensors()
input_details = clf_interpreter.get_input_details()
output_details = clf_interpreter.get_output_details()

THRESH = 0.50  # Threshold to decide facing

cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)

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

    # Determine facing
    facing = prob > THRESH
    label = "Not Facing" if facing else "Facing"
    color = (0, 0, 255) if facing else (0, 255, 0)

    # Draw label on original frame
    cv2.putText(frame, f"{label} ({prob:.2f})", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("MobileNet Facing Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
