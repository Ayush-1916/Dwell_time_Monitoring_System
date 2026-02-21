import cv2
import numpy as np
import tensorflow as tf

# ---------------- Load the classifier ----------------
clf_interpreter = tf.lite.Interpreter(model_path="facing_classifier.tflite")
clf_interpreter.allocate_tensors()
clf_input_details = clf_interpreter.get_input_details()
clf_output_details = clf_interpreter.get_output_details()

# ---------------- Load the test image ----------------
img_path = r"C:\A_Dwell_time\dataset\train\facing\000015.jpg"  # Replace with your image path
img = cv2.imread(img_path)
if img is None:
    raise ValueError("Image not found or path is incorrect")

# Resize to classifier input size
input_h, input_w = clf_input_details[0]['shape'][1:3]
img_resized = cv2.resize(img, (input_w, input_h))
input_data = np.expand_dims(img_resized.astype(np.float32)/255.0, axis=0)

# ---------------- Run inference ----------------
clf_interpreter.set_tensor(clf_input_details[0]['index'], input_data)
clf_interpreter.invoke()
prob = clf_interpreter.get_tensor(clf_output_details[0]['index'])[0][0]

print(f"Facing probability: {prob:.4f}")
print("Prediction:", "Not Facing" if prob > 0.4 else "Facing")

# Optional: show image
cv2.imshow("Test Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
